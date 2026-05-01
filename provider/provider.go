package provider

import (
	"context"

	"github.com/teoclub/seedpod/schema"
)

// Provider describes a backend capable of generating responses from LLMs.
type Provider interface {
	Name() string
	Generate(ctx context.Context, prompt []*schema.Message, opts ...Option) (*schema.Response, error)
	Stream(ctx context.Context, prompt []*schema.Message, opts ...Option) (<-chan schema.StreamChunk, error)
}
