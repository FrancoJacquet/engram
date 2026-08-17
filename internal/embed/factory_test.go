package embed

import (
	"errors"
	"testing"
)

func TestEmptyCommandReturnsErrNoEmbedder(t *testing.T) {
	_, err := NewEmbedder("")
	if !errors.Is(err, ErrNoEmbedder) {
		t.Fatalf("expected ErrNoEmbedder, got %v", err)
	}
}
