package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/teoclub/seedpod"
	"github.com/teoclub/seedpod/internal/httpretry"
	"github.com/teoclub/seedpod/internal/sse"
	"github.com/teoclub/seedpod/provider"
	"github.com/teoclub/seedpod/schema"
)

const (
	providerName     = "anthropic"
	defaultBaseURL   = "https://api.anthropic.com"
	messagesPath     = "/v1/messages"
	versionHeader    = "2023-06-01"
	defaultModel     = "claude-3-haiku-20240307"
	defaultMaxTokens = 1024
)

type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	Thinking string       `json:"thinking,omitempty"`
	Source   *imageSource `json:"source,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicResponse struct {
	ID      string             `json:"id"`
	Content []anthropicContent `json:"content"`
	Usage   usageBlock         `json:"usage"`
}

type usageBlock struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicDeltaEvent struct {
	Delta struct {
		Text     string `json:"text"`
		Thinking string `json:"thinking"`
	} `json:"delta"`
}

type anthropicErrorEvent struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
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
		return nil, fmt.Errorf("anthropic: %w: api key is required", seedpod.ErrInvalidInput)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 2 * time.Minute}
	}
	return &Client{baseCfg: cfg}, nil
}

func (c *Client) Name() string { return providerName }

func (c *Client) Generate(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
	cfg := c.mergeConfig(opts...)
	payload, err := buildRequestPayload(prompt, cfg, false)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+messagesPath, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		applyHeaders(req, cfg)
		return req, nil
	}, httpretry.DefaultConfig())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	parts, err := convertResponseContent(decoded.Content)
	if err != nil {
		return nil, err
	}
	return &schema.Response{
		ID:      decoded.ID,
		Content: parts,
		Usage: schema.UsageMetadata{
			PromptTokens:     decoded.Usage.InputTokens,
			CompletionTokens: decoded.Usage.OutputTokens,
			TotalTokens:      decoded.Usage.InputTokens + decoded.Usage.OutputTokens,
		},
		Raw: decoded,
	}, nil
}

func (c *Client) Stream(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error) {
	cfg := c.mergeConfig(opts...)
	payload, err := buildRequestPayload(prompt, cfg, true)
	if err != nil {
		return nil, err
	}
	resp, err := httpretry.Do(ctx, cfg.HTTPClient, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+messagesPath, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		applyHeaders(req, cfg)
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
			ch <- schema.StreamChunk{Err: fmt.Errorf("anthropic: http %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), Done: true}
			return
		}

		decoder := sse.NewDecoder(resp.Body)
		for {
			event, err := decoder.Next()
			if err != nil {
				if errors.Is(err, io.EOF) {
					return
				}
				ch <- schema.StreamChunk{Err: err, Done: true}
				return
			}
			switch event.Name {
			case "content_block_delta":
				var delta anthropicDeltaEvent
				if err := json.Unmarshal([]byte(event.Data), &delta); err != nil {
					ch <- schema.StreamChunk{Err: err, Done: true}
					return
				}
				text := strings.TrimSpace(delta.Delta.Text)
				reasoning := strings.TrimSpace(delta.Delta.Thinking)
				if text == "" && reasoning == "" {
					continue
				}
				select {
				case <-ctx.Done():
					ch <- schema.StreamChunk{Err: ctx.Err(), Done: true}
					return
				case ch <- schema.StreamChunk{Delta: text, ReasoningDelta: reasoning}:
				}
			case "message_stop":
				ch <- schema.StreamChunk{Done: true}
				return
			case "error":
				var payload anthropicErrorEvent
				if err := json.Unmarshal([]byte(event.Data), &payload); err != nil {
					ch <- schema.StreamChunk{Err: err, Done: true}
				} else {
					ch <- schema.StreamChunk{Err: errors.New(payload.Error.Message), Done: true}
				}
				return
			default:
				if event.Data == "[DONE]" {
					ch <- schema.StreamChunk{Done: true}
					return
				}
			}
		}
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
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = c.baseCfg.MaxTokens
	}
	return cfg
}

func buildRequestPayload(prompt []*schema.Message, cfg provider.Config, stream bool) ([]byte, error) {
	messages := make([]anthropicMessage, 0, len(prompt))
	var systemParts []string
	for _, msg := range prompt {
		if msg == nil {
			continue
		}
		if msg.Role == schema.RoleSystem {
			text, err := concatTextContent(msg.Content)
			if err != nil {
				return nil, err
			}
			if text != "" {
				systemParts = append(systemParts, text)
			}
			continue
		}
		converted, err := convertToAnthropicMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	req := anthropicRequest{
		Model:       cfg.Model,
		Messages:    messages,
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		Stream:      stream,
	}
	if len(systemParts) > 0 {
		req.System = strings.Join(systemParts, "\n\n")
	}
	return json.Marshal(req)
}

func applyHeaders(r *http.Request, cfg provider.Config) {
	r.Header.Set("x-api-key", cfg.APIKey)
	r.Header.Set("anthropic-version", versionHeader)
	r.Header.Set("content-type", "application/json")
	for k, v := range cfg.Headers {
		r.Header.Set(k, v)
	}
}

func convertToAnthropicMessage(msg *schema.Message) (anthropicMessage, error) {
	if msg == nil {
		return anthropicMessage{}, fmt.Errorf("anthropic: %w: message is nil", seedpod.ErrInvalidInput)
	}
	parts := make([]anthropicContent, 0, len(msg.Content))
	for _, part := range msg.Content {
		switch v := part.(type) {
		case *schema.TextContent:
			parts = append(parts, anthropicContent{Type: "text", Text: v.Text})
		case *schema.ImageContent:
			source, err := imageSourceFromContent(v)
			if err != nil {
				return anthropicMessage{}, err
			}
			parts = append(parts, anthropicContent{Type: "image", Source: source})
		default:
			return anthropicMessage{}, fmt.Errorf("anthropic: unsupported content type %T", v)
		}
	}
	if len(parts) == 0 {
		parts = append(parts, anthropicContent{Type: "text", Text: ""})
	}
	return anthropicMessage{Role: string(msg.Role), Content: parts}, nil
}

func concatTextContent(parts []schema.ContentPart) (string, error) {
	var b strings.Builder
	for _, part := range parts {
		text, ok := part.(*schema.TextContent)
		if !ok {
			return "", fmt.Errorf("anthropic: system messages must be text only")
		}
		b.WriteString(text.Text)
	}
	return strings.TrimSpace(b.String()), nil
}

func imageSourceFromContent(img *schema.ImageContent) (*imageSource, error) {
	if img == nil {
		return nil, fmt.Errorf("anthropic: %w: image is nil", seedpod.ErrInvalidInput)
	}
	if strings.HasPrefix(img.URL, "data:") {
		media, data, err := parseDataURL(img.URL)
		if err != nil {
			return nil, err
		}
		return &imageSource{Type: "base64", MediaType: media, Data: data}, nil
	}
	return &imageSource{Type: "url", URL: img.URL}, nil
}

func parseDataURL(u string) (string, string, error) {
	if !strings.HasPrefix(u, "data:") {
		return "", "", fmt.Errorf("anthropic: not a data url")
	}
	payload := strings.TrimPrefix(u, "data:")
	parts := strings.SplitN(payload, ",", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("anthropic: malformed data url")
	}
	meta, data := parts[0], parts[1]
	mediaType := "application/octet-stream"
	if meta != "" {
		fields := strings.Split(meta, ";")
		mediaType = fields[0]
		foundBase64 := false
		for _, f := range fields[1:] {
			if f == "base64" {
				foundBase64 = true
				break
			}
		}
		if !foundBase64 {
			return "", "", fmt.Errorf("anthropic: data urls must be base64 encoded")
		}
	}
	if data == "" {
		return "", "", fmt.Errorf("anthropic: data url is empty")
	}
	return mediaType, data, nil
}

func convertResponseContent(blocks []anthropicContent) ([]schema.ContentPart, error) {
	parts := make([]schema.ContentPart, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, schema.Text(block.Text))
		case "thinking", "reasoning":
			parts = append(parts, schema.Reasoning(firstNonEmpty(block.Thinking, block.Text)))
		case "image":
			if block.Source != nil {
				url := block.Source.URL
				if url == "" && block.Source.Data != "" {
					url = fmt.Sprintf("data:%s;base64,%s", block.Source.MediaType, block.Source.Data)
				}
				parts = append(parts, &schema.ImageContent{URL: url})
			}
		}
	}
	return parts, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
