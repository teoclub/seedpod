package seedpod

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/teoclub/seedpod/provider"
	"github.com/teoclub/seedpod/schema"
)

type GenerateFn func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error)
type StreamFn func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error)

type testProvider struct {
	name string

	generateFn GenerateFn
	streamFn   StreamFn
}

func NewTestProvider(name string, generateFn GenerateFn, streamFn StreamFn) *testProvider {
	return &testProvider{
		name:       name,
		generateFn: generateFn,
		streamFn:   streamFn,
	}
}

func (p *testProvider) Name() string {
	if p.name != "" {
		return p.name
	}
	return "test"
}

func (p *testProvider) Generate(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
	if p.generateFn != nil {
		return p.generateFn(ctx, prompt, opts...)
	}
	return &schema.Response{Content: []schema.ContentPart{schema.Text("default")}}, nil
}

func (p *testProvider) Stream(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error) {
	if p.streamFn != nil {
		return p.streamFn(ctx, prompt, opts...)
	}
	ch := make(chan schema.StreamChunk, 1)
	ch <- schema.StreamChunk{Delta: "default", Done: true}
	close(ch)
	return ch, nil
}

func TestClientGenerate(t *testing.T) {
	var captured bool
	stub := NewTestProvider("test", func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
		captured = true
		if len(prompt) != 1 {
			t.Fatalf("expected prompt length 1")
		}
		cfg := provider.NewConfig(opts...)
		if cfg.Model != "unit-test" {
			t.Fatalf("expected option propagation")
		}
		return &schema.Response{ID: "1", Content: []schema.ContentPart{schema.Text("ok")}}, nil
	}, nil)

	client := Wrap(stub, provider.WithModel("unit-test"))
	resp, err := client.Generate(context.Background(), []*schema.Message{schema.NewUserMessage(schema.Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Text() != "ok" || !captured {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestClientGenerateCostCalculation(t *testing.T) {
	stub := NewTestProvider("cost-test", func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
		return &schema.Response{
			ID:      "cost-test",
			Content: []schema.ContentPart{schema.Text("ok")},
			Usage: schema.UsageMetadata{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
			},
		}, nil
	}, nil)

	client := Wrap(stub, provider.WithModel("unit-test"), provider.WithCost(2.50, 10.00))
	resp, err := client.Generate(context.Background(), []*schema.Message{schema.NewUserMessage(schema.Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	// Expected: (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
	//         = 0.0025 + 0.005 = 0.0075
	expected := 0.0075
	if math.Abs(resp.Usage.Cost-expected) > 1e-9 {
		t.Fatalf("expected cost %f, got %f", expected, resp.Usage.Cost)
	}
}

func TestClientGenerateCostZeroWhenNotConfigured(t *testing.T) {
	stub := NewTestProvider("no-cost-test", func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
		return &schema.Response{
			ID:      "no-cost",
			Content: []schema.ContentPart{schema.Text("ok")},
			Usage: schema.UsageMetadata{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
			},
		}, nil
	}, nil)

	client := Wrap(stub)
	resp, err := client.Generate(context.Background(), []*schema.Message{schema.NewUserMessage(schema.Text("hi"))})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	if resp.Usage.Cost != 0 {
		t.Fatalf("expected zero cost when not configured, got %f", resp.Usage.Cost)
	}
}

func TestClientGenerateCostPerRequestOverride(t *testing.T) {
	stub := NewTestProvider("override-test", func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
		return &schema.Response{
			ID:      "override",
			Content: []schema.ContentPart{schema.Text("ok")},
			Usage: schema.UsageMetadata{
				PromptTokens:     2000,
				CompletionTokens: 1000,
				TotalTokens:      3000,
			},
		}, nil
	}, nil)

	// Client defaults: $5.00/$15.00 per 1M tokens
	client := Wrap(stub, provider.WithCost(5.00, 15.00))
	// Per-request override: $1.00/$3.00 per 1M tokens
	resp, err := client.Generate(context.Background(),
		[]*schema.Message{schema.NewUserMessage(schema.Text("hi"))},
		provider.WithCost(1.00, 3.00),
	)
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}
	// Expected: (2000 * 1.00 / 1_000_000) + (1000 * 3.00 / 1_000_000)
	//         = 0.002 + 0.003 = 0.005
	expected := 0.005
	if math.Abs(resp.Usage.Cost-expected) > 1e-9 {
		t.Fatalf("expected cost %f, got %f", expected, resp.Usage.Cost)
	}
}

func TestClientGeneratePropagatesErrors(t *testing.T) {
	stub := NewTestProvider("error-test", func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
		return nil, errors.New("boom")
	}, nil)

	client := Wrap(stub)
	if _, err := client.Generate(context.Background(), nil); err == nil {
		t.Fatalf("expected error")
	}
}

func TestClientStream(t *testing.T) {
	stub := NewTestProvider("stream-test", nil, func(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error) {
		ch := make(chan schema.StreamChunk, 2)
		ch <- schema.StreamChunk{Delta: "hello "}
		ch <- schema.StreamChunk{Delta: "world", Done: true}
		close(ch)
		return ch, nil
	})

	client := Wrap(stub)
	stream, err := client.Stream(context.Background(), nil)
	if err != nil {
		t.Fatalf("stream failed: %v", err)
	}
	var combined strings.Builder
	for chunk := range stream {
		if chunk.Err != nil {
			t.Fatalf("chunk error: %v", chunk.Err)
		}
		combined.WriteString(chunk.Delta)
		if chunk.Done {
			break
		}
	}
	if combined.String() != "hello world" {
		t.Fatalf("unexpected stream result: %s", combined.String())
	}
}
