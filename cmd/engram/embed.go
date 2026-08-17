package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/Gentleman-Programming/engram/internal/embed"
	"github.com/Gentleman-Programming/engram/internal/store"
)

// defaultEmbedBatch matches the embedder's safe batch bound: bigger batches
// blew the ONNX arena up to ~9 GB on real content and the OOM killer took
// the whole session down.
const defaultEmbedBatch = 16

func normalizeBatch(n int) int {
	if n <= 0 {
		return defaultEmbedBatch
	}
	return n
}

// cmdEmbed implements `engram embed --backfill`.
func cmdEmbed(cfg store.Config) {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	backfill := fs.Bool("backfill", false, "embed observations that are missing vectors")
	batch := fs.Int("batch", defaultEmbedBatch, "observations per batch")
	_ = fs.Parse(os.Args[2:])
	if !*backfill {
		fmt.Fprintln(os.Stderr, "usage: engram embed --backfill [--batch N]")
		exitFunc(1)
		return
	}

	s, err := storeNew(cfg)
	if err != nil {
		fatal(err)
	}
	defer s.Close()

	e, err := embed.NewEmbedder(os.Getenv("ENGRAM_EMBEDDER"))
	if err != nil {
		fatal(err)
	}
	defer e.Close()

	ctx := context.Background()
	models, err := e.Models(ctx)
	if err != nil {
		fatal(fmt.Errorf("embedder handshake failed: %w", err))
	}

	total, failed := 0, 0
	for _, m := range models {
		fmt.Printf("model %s (dim %d):\n", m.Name, m.Dim)
		for {
			pend, err := s.PendingEmbeddings(m.Name, normalizeBatch(*batch))
			if err != nil {
				fatal(err)
			}
			if len(pend) == 0 {
				break
			}

			reqs := make([]embed.Request, 0, len(pend))
			for _, p := range pend {
				reqs = append(reqs, embed.Request{
					ID:    strconv.FormatInt(p.ID, 10),
					Kind:  embed.KindPassage,
					Title: p.Title,
					Text:  p.Content,
				})
			}

			res, err := e.Embed(ctx, reqs)
			if err != nil {
				fatal(fmt.Errorf("embedder failed: %w", err))
			}

			// Persist every model's vector from this call, not just m's: the
			// next model's outer pass then finds almost nothing pending.
			progressed := false
			for _, r := range res {
				if r.Err != nil {
					failed++
					fmt.Fprintf(os.Stderr, "obs %s (%s): %v\n", r.ID, r.Model, r.Err)
					continue
				}
				id, convErr := strconv.ParseInt(r.ID, 10, 64)
				if convErr != nil {
					failed++
					continue
				}
				if err := s.SetEmbedding(id, r.Model, r.Vector); err != nil {
					fatal(err)
				}
				total++
				if r.Model == m.Name {
					progressed = true
				}
			}

			// If a whole batch failed for the current model, PendingEmbeddings
			// would return the same rows forever. Stop instead of spinning.
			if !progressed {
				fatal(fmt.Errorf("entire batch failed for %s (%d errors so far)", m.Name, failed))
			}
			fmt.Printf("  stored %d vectors…\n", total)
		}
	}

	fmt.Printf("done: %d vectors stored, %d failed\n", total, failed)
}
