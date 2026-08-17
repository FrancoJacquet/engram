package store

import (
	"context"
	"os"
	"strconv"
	"time"

	"github.com/Gentleman-Programming/engram/internal/embed"
)

// asyncEmbedTimeout bounds one background embedding. Generous on purpose:
// the first call after a cold start pays the model load.
const asyncEmbedTimeout = 60 * time.Second

// embedAsync embeds a freshly saved observation in the background.
// Best-effort by design: if the embedder is missing, fails or hangs, the
// observation simply has no vectors yet and the next
// `engram embed --backfill` picks it up. Saving a memory never fails or
// stalls because of a model.
func (s *Store) embedAsync(id int64, title, content string) {
	cmd := os.Getenv("ENGRAM_EMBEDDER")
	if cmd == "" {
		return
	}

	go func() {
		defer func() { _ = recover() }() // never take the process down for this

		e, err := s.sharedEmbedder(cmd)
		if err != nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), asyncEmbedTimeout)
		defer cancel()

		res, err := e.Embed(ctx, []embed.Request{{
			ID:    strconv.FormatInt(id, 10),
			Kind:  embed.KindPassage,
			Title: title,
			Text:  content,
		}})
		if err != nil {
			return
		}
		for _, r := range res {
			if r.Err != nil || len(r.Vector) == 0 {
				continue
			}
			_ = s.SetEmbedding(id, r.Model, r.Vector)
		}
	}()
}
