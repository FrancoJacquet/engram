package main

import "testing"

func TestBatchSizeHasAFloor(t *testing.T) {
	if got := normalizeBatch(0); got != defaultEmbedBatch {
		t.Errorf("batch 0 -> %d, expected %d", got, defaultEmbedBatch)
	}
	if got := normalizeBatch(-5); got != defaultEmbedBatch {
		t.Errorf("negative batch -> %d, expected %d", got, defaultEmbedBatch)
	}
	if got := normalizeBatch(8); got != 8 {
		t.Errorf("batch 8 -> %d, expected 8", got)
	}
}
