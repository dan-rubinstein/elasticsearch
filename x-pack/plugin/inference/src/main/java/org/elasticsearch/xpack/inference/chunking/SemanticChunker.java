/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.chunking;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.ChunkedInference;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.InferenceService;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.elasticsearch.xpack.inference.services.elasticsearch.ElasticsearchInternalService.EMBEDDING_MAX_BATCH_SIZE;

public class SemanticChunker {
    private ActionListener<List<ChunkedInference>> finalListener;
    private ActionListener<List<EmbeddingRequestChunker.BatchRequestAndListener>> batchedRequestsListener;
    private InferenceService inferenceService;
    private Model model;
    private final ChunkingSettings chunkingSettings;
    private final AtomicInteger resultCount = new AtomicInteger();
    private List<SemanticChunkerUtils.SemanticChunkData> chunks;
    private final String input;

    public SemanticChunker(
        InferenceService inferenceService,
        Model model,
        String input,
        ChunkingSettings chunkingSettings,
        ActionListener<List<EmbeddingRequestChunker.BatchRequestAndListener>> batchedRequestsListener,
        ActionListener<List<ChunkedInference>> finalListener
    ) {
        this.inferenceService = inferenceService;
        this.model = model;
        this.input = input;
        this.chunkingSettings = chunkingSettings;
        this.batchedRequestsListener = batchedRequestsListener; // TODO: Figure out how to run the batched requests.
        this.finalListener = finalListener;
    }

    public void chunk() {
        chunks = SemanticChunkerUtils.chunk(input, chunkingSettings);

        for (var chunk : chunks) {
            inferenceService.infer(
                model,
                null,
                List.of(chunk.combinedSentence),
                false,
                null,
                null,
                TimeValue.ONE_HOUR,
                new DebatchingListener(chunk, chunks.size())
            );
        }
    }

    private void postInference() {
        calculateCosineDistanceToNextForChunks();
        var chunksWithValidCosineDistanceDiff = getChunksWithValidCosineDistanceDiff();

        List<EmbeddingRequestChunker.BatchRequestAndListener> batchedRequests = new EmbeddingRequestChunker(
            List.of(input), // TODO: Change this to just input when fixing the input type
            List.of(chunksWithValidCosineDistanceDiff),
            EMBEDDING_MAX_BATCH_SIZE
        ).batchRequestsWithListeners(finalListener);

        batchedRequestsListener.onResponse(batchedRequests);
    }

    private void calculateCosineDistanceToNextForChunks() {
        for (int i = 0; i < chunks.size() - 1; i++) {
            chunks.get(i)
                .setCosineDistanceToNext(
                    EmbeddingResultUtils.cosineDistance(chunks.get(i).inferenceServiceResults, chunks.get(i + 1).inferenceServiceResults)
                );
        }
    }

    private List<Chunker.ChunkOffset> getChunksWithValidCosineDistanceDiff() {
        int startOfGroupIndex = 0;
        List<Chunker.ChunkOffset> chunkOffsets = new ArrayList<>();
        var threshold = chunkingSettings instanceof SemanticChunkingSettings
            ? ((SemanticChunkingSettings) chunkingSettings).threshold
            : 0.95f;
        var thresholdValue = EmbeddingResultUtils.getPercentile(
            chunks.stream().map(chunk -> chunk.cosineDistanceToNext).toList(),
            threshold
        );
        for (int i = 0; i < chunks.size(); i++) {
            if (i == chunks.size() - 1 || chunks.get(i).cosineDistanceToNext > thresholdValue) {
                chunkOffsets.add(
                    new Chunker.ChunkOffset(chunks.get(startOfGroupIndex).chunkOffset.start(), chunks.get(i).chunkOffset.end())
                );
                startOfGroupIndex = i + 1;
            }
        }

        return chunkOffsets;
    }

    private class DebatchingListener implements ActionListener<InferenceServiceResults> {
        // TODO: Can maybe remove this in favor of embeddingrequestchunker now that we added a new function there
        private final SemanticChunkerUtils.SemanticChunkData chunk;
        private final int batchSize;

        private DebatchingListener(SemanticChunkerUtils.SemanticChunkData chunk, int batchSize) {
            this.chunk = chunk;
            this.batchSize = batchSize;
        }

        @Override
        public void onResponse(InferenceServiceResults inferenceServiceResults) {
            chunk.setInferenceServiceResults(inferenceServiceResults);
            if (resultCount.incrementAndGet() == batchSize) {
                postInference();
            }
        }

        @Override
        public void onFailure(Exception e) {
            finalListener.onFailure(e);
        }
    }
}
