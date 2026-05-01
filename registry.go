package seedpod

import (
	"fmt"
	"sort"
	"sync"

	"github.com/teoclub/seedpod/provider"
)

// ProviderFactory describes a function that can instantiate a provider backed by a specific vendor SDK.
type ProviderFactory func(apiKey string, opts ...provider.Option) (provider.Provider, error)

var (
	registryMu sync.RWMutex
	factories  = map[string]ProviderFactory{}
)

// RegisterProvider adds a provider factory to the global registry.
func RegisterProvider(name string, factory ProviderFactory) error {
	if name == "" {
		return fmt.Errorf("%w: name is empty", ErrInvalidInput)
	}
	if factory == nil {
		return fmt.Errorf("%w: factory is nil", ErrInvalidInput)
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := factories[name]; exists {
		return fmt.Errorf("%w: %s", ErrProviderAlreadyRegistered, name)
	}
	factories[name] = factory
	return nil
}

// MustRegisterProvider registers a provider and panics on failure.
func MustRegisterProvider(name string, factory ProviderFactory) {
	if err := RegisterProvider(name, factory); err != nil {
		panic(err)
	}
}

// RegisteredProviders returns a sorted slice of registered provider names.
func RegisteredProviders() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(factories))
	for name := range factories {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// lookupProvider fetches the factory for a provider name.
func lookupProvider(name string) (ProviderFactory, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	factory, ok := factories[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrProviderNotFound, name)
	}
	return factory, nil
}

// New creates a Client bound to the named provider.
func New(providerName, apiKey string, opts ...provider.Option) (*Client, error) {
	factory, err := lookupProvider(providerName)
	if err != nil {
		return nil, err
	}
	provider, err := factory(apiKey, opts...)
	if err != nil {
		return nil, err
	}
	return Wrap(provider, opts...), nil
}

// MustNew is a helper that panics if client creation fails.
func MustNew(providerName, apiKey string, opts ...provider.Option) *Client {
	client, err := New(providerName, apiKey, opts...)
	if err != nil {
		panic(err)
	}
	return client
}
