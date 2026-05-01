package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/teoclub/seedpod"
	"github.com/teoclub/seedpod/internal/httpretry"
	"github.com/teoclub/seedpod/provider"
	"github.com/teoclub/seedpod/schema"
)

const (
	providerName      = "gemini"
	defaultBaseURL    = "https://generativelanguage.googleapis.com"
	defaultModel      = "gemini-1.5-flash"
	generateSuffix    = ":generateContent"
	streamSuffix      = ":streamGenerateContent"
	apiVersionSegment = "v1beta"
)

type geminiRequest struct {
	Contents          []geminiContent   `json:"contents"`
	SystemInstruction *geminiContent    `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool      `json:"tools,omitempty"`
}

type geminiTool struct {
	GoogleSearch *googleSearchTool `json:"googleSearch,omitempty"`
}

type googleSearchTool struct{}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text       string      `json:"text,omitempty"`
	Thought    bool        `json:"thought,omitempty"`
	InlineData *inlineData `json:"inlineData,omitempty"`
	FileData   *fileData   `json:"fileData,omitempty"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	FileURI string `json:"fileUri"`
}

type generationConfig struct {
	Temperature        float64  `json:"temperature,omitempty"`
	MaxOutputTokens    int      `json:"maxOutputTokens,omitempty"`
	ResponseModalities []string `json:"responseModalities,omitempty"`
}

type geminiResponse struct {
	Candidates []candidate   `json:"candidates"`
	Usage      usageMetadata `json:"usageMetadata"`
}

type candidate struct {
	Content geminiContent `json:"content"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type Client struct {
	baseCfg provider.Config
}

func init() { seedpod.MustRegisterProvider(providerName, New) }

func New(apiKey string, opts ...provider.Option) (provider.Provider, error) {
	cfg := provider.NewConfig(opts...)
	if cfg.APIKey == "" {
		cfg.APIKey = apiKey
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gemini: %w: api key is required", seedpod.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	return &Client{baseCfg: cfg}, nil
}

func (c *Client) Name() string { return providerName }

func (c *Client) Generate(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildRequestBody(prompt, cfg)
	if err != nil {
		return nil, err
	}
	endpoint, err := buildEndpoint(cfg, false)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("content-type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		return req, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	parts, err := convertCandidate(decoded.Candidates)
	if err != nil {
		return nil, err
	}
	usage := decoded.Usage
	return &schema.Response{
		Content: parts,
		Usage: schema.UsageMetadata{
			PromptTokens:     usage.PromptTokenCount,
			CompletionTokens: usage.CandidatesTokenCount,
			TotalTokens:      usage.TotalTokenCount,
		},
		Raw: decoded,
	}, nil
}

func (c *Client) Stream(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error) {
	cfg := c.mergeConfig(opts...)
	reqBody, err := buildRequestBody(prompt, cfg)
	if err != nil {
		return nil, err
	}
	endpoint, err := buildEndpoint(cfg, true)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("content-type", "application/json")
		for k, v := range cfg.Headers {
			req.Header.Set(k, v)
		}
		return req, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}

	ch := make(chan schema.StreamChunk)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(resp.Body)
			ch <- schema.StreamChunk{Err: fmt.Errorf("gemini: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), Done: true}
			return
		}
		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 4*1024*1024)
		var pending strings.Builder
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "data:") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
			if line == "" || line == "[DONE]" {
				continue
			}
			pending.WriteString(line)
			payload := pending.String()
			var parsed []geminiResponse
			var chunk geminiResponse
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				if strings.Contains(err.Error(), "unexpected end of JSON input") {
					continue
				}
				if err := json.Unmarshal([]byte(payload), &parsed); err != nil {
					ch <- schema.StreamChunk{Err: err, Done: true}
					return
				}
			} else {
				parsed = []geminiResponse{chunk}
			}
			pending.Reset()
			for _, current := range parsed {
				text, reasoning, err := extractContent(current.Candidates)
				if err != nil {
					ch <- schema.StreamChunk{Err: err, Done: true}
					return
				}
				if text == "" && reasoning == "" {
					continue
				}
				select {
				case <-ctx.Done():
					ch <- schema.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- schema.StreamChunk{Delta: text, ReasoningDelta: reasoning}:
				}
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- schema.StreamChunk{Err: err, Done: true}
			return
		}
		ch <- schema.StreamChunk{Done: true}
	}()

	return ch, nil
}

func (c *Client) mergeConfig(opts ...provider.Option) provider.Config {
	cfg := c.baseCfg.Clone()
	provider.Apply(&cfg, opts...)
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = c.baseCfg.HTTPClient
	}
	if cfg.Model == "" {
		cfg.Model = c.baseCfg.Model
	}
	if cfg.APIKey == "" {
		cfg.APIKey = c.baseCfg.APIKey
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = c.baseCfg.BaseURL
	}
	return cfg
}

func buildEndpoint(cfg provider.Config, stream bool) (string, error) {
	base, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return "", err
	}
	suffix := generateSuffix
	if stream {
		suffix = streamSuffix
	}
	base.Path = path.Join(strings.TrimSuffix(base.Path, "/"), apiVersionSegment, "models", cfg.Model) + suffix
	q := base.Query()
	q.Set("key", cfg.APIKey)
	base.RawQuery = q.Encode()
	return base.String(), nil
}

func buildRequestBody(prompt []*schema.Message, cfg provider.Config) ([]byte, error) {
	contents, system, err := convertMessages(prompt)
	if err != nil {
		return nil, err
	}
	req := geminiRequest{
		Contents: contents,
	}
	if system != nil {
		req.SystemInstruction = system
	}
	if cfg.Temperature != 0 || cfg.MaxTokens != 0 || len(cfg.ResponseModalities) > 0 {
		req.GenerationConfig = &generationConfig{}
		if cfg.Temperature != 0 {
			req.GenerationConfig.Temperature = cfg.Temperature
		}
		if cfg.MaxTokens != 0 {
			req.GenerationConfig.MaxOutputTokens = cfg.MaxTokens
		}
		if len(cfg.ResponseModalities) > 0 {
			req.GenerationConfig.ResponseModalities = cfg.ResponseModalities
		}
	}
	if cfg.EnableWebSearch {
		req.Tools = []geminiTool{
			{GoogleSearch: &googleSearchTool{}},
		}
	}
	return json.Marshal(req)
}

func convertMessages(prompt []*schema.Message) ([]geminiContent, *geminiContent, error) {
	contents := make([]geminiContent, 0, len(prompt))
	var systemParts []geminiPart
	for _, msg := range prompt {
		if msg == nil {
			continue
		}
		switch msg.Role {
		case schema.RoleSystem:
			parts, err := convertParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			systemParts = append(systemParts, parts...)
		default:
			parts, err := convertParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}
			if len(parts) == 0 {
				continue
			}
			contents = append(contents, geminiContent{Role: string(msg.Role), Parts: parts})
		}
	}
	var system *geminiContent
	if len(systemParts) > 0 {
		system = &geminiContent{Parts: systemParts}
	}
	return contents, system, nil
}

func convertParts(parts []schema.ContentPart) ([]geminiPart, error) {
	converted := make([]geminiPart, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case *schema.TextContent:
			converted = append(converted, geminiPart{Text: v.Text})
		case *schema.ImageContent:
			inline, err := toInlineData(v.URL)
			if err != nil {
				// Fall back to file reference if not inline.
				converted = append(converted, geminiPart{FileData: &fileData{FileURI: v.URL}})
			} else {
				converted = append(converted, geminiPart{InlineData: inline})
			}
		default:
			return nil, fmt.Errorf("gemini: unsupported content type %T", v)
		}
	}
	return converted, nil
}

func toInlineData(urlStr string) (*inlineData, error) {
	if !strings.HasPrefix(urlStr, "data:") {
		return nil, fmt.Errorf("not inline data")
	}
	payload := strings.TrimPrefix(urlStr, "data:")
	parts := strings.SplitN(payload, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("gemini: malformed data url")
	}
	meta, data := parts[0], parts[1]
	mime := "application/octet-stream"
	if meta != "" {
		fields := strings.Split(meta, ";")
		mime = fields[0]
		encoded := false
		for _, f := range fields[1:] {
			if f == "base64" {
				encoded = true
				break
			}
		}
		if !encoded {
			return nil, fmt.Errorf("gemini: data urls must be base64 encoded")
		}
	}
	if data == "" {
		return nil, fmt.Errorf("gemini: data url empty")
	}
	return &inlineData{MimeType: mime, Data: data}, nil
}

func convertCandidate(candidates []candidate) ([]schema.ContentPart, error) {
	if len(candidates) == 0 {
		return nil, errors.New("gemini: no candidates returned")
	}
	return convertGeminiParts(candidates[0].Content.Parts)
}

func convertGeminiParts(parts []geminiPart) ([]schema.ContentPart, error) {
	out := make([]schema.ContentPart, 0, len(parts))
	for _, part := range parts {
		switch {
		case part.Text != "" && part.Thought:
			out = append(out, schema.Reasoning(part.Text))
		case part.Text != "":
			out = append(out, schema.Text(part.Text))
		case part.InlineData != nil:
			dataURL := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data)
			out = append(out, &schema.ImageContent{URL: dataURL})
		case part.FileData != nil:
			out = append(out, &schema.ImageContent{URL: part.FileData.FileURI})
		}
	}
	return out, nil
}

func extractContent(candidates []candidate) (string, string, error) {
	parts, err := convertCandidate(candidates)
	if err != nil {
		return "", "", err
	}
	var textBuilder strings.Builder
	var reasoningBuilder strings.Builder
	for _, part := range parts {
		switch value := part.(type) {
		case *schema.TextContent:
			textBuilder.WriteString(value.Text)
		case *schema.ReasoningContent:
			reasoningBuilder.WriteString(value.Text)
		}
	}
	return textBuilder.String(), reasoningBuilder.String(), nil
}
