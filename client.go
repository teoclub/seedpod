package seedpod

import (
	"context"

	"github.com/teoclub/seedpod/provider"
	"github.com/teoclub/seedpod/schema"
)

// Client is the main entry point for interacting with LLM providers via the unified API.
type Client struct {
	provider    provider.Provider
	defaultOpts []provider.Option
}

// Wrap binds an already-constructed provider to a Client.
func Wrap(p provider.Provider, opts ...provider.Option) *Client {
	return &Client{
		provider:    p,
		defaultOpts: append([]provider.Option(nil), opts...),
	}
}

// ProviderName returns the underlying provider identifier.
func (c *Client) ProviderName() string {
	if c == nil || c.provider == nil {
		return ""
	}
	return c.provider.Name()
}

// Generate performs a single request/response interaction with the provider.
func (c *Client) Generate(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (*schema.Response, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Generate", Err: ErrInvalidInput}
	}
	merged := provider.MergeOptions(c.defaultOpts, opts)
	resp, err := c.provider.Generate(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Generate", Err: err}
	}
	computeCost(resp, merged)
	return resp, nil
}

// computeCost populates Response.Usage.Cost based on token counts and configured rates.
func computeCost(resp *schema.Response, opts []provider.Option) {
	if resp == nil {
		return
	}
	cfg := provider.NewConfig(opts...)
	inputCost := float64(resp.Usage.PromptTokens) * cfg.InputCostPerMillionTokens / 1_000_000
	outputCost := float64(resp.Usage.CompletionTokens) * cfg.OutputCostPerMillionTokens / 1_000_000
	resp.Usage.Cost = inputCost + outputCost
}

// Stream initiates a streaming interaction and returns a read-only channel of chunks.
func (c *Client) Stream(ctx context.Context, prompt []*schema.Message, opts ...provider.Option) (<-chan schema.StreamChunk, error) {
	if c == nil || c.provider == nil {
		return nil, &Error{Provider: "", Op: "Stream", Err: ErrInvalidInput}
	}
	merged := provider.MergeOptions(c.defaultOpts, opts)
	ch, err := c.provider.Stream(ctx, prompt, merged...)
	if err != nil {
		return nil, &Error{Provider: c.provider.Name(), Op: "Stream", Err: err}
	}
	return ch, nil
}
