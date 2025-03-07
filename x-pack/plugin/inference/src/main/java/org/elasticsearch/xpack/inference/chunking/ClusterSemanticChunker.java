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
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.elasticsearch.xpack.inference.services.elasticsearch.ElasticsearchInternalService.EMBEDDING_MAX_BATCH_SIZE;

public class ClusterSemanticChunker {
    private ActionListener<List<ChunkedInference>> finalListener;
    private ActionListener<List<EmbeddingRequestChunker.BatchRequestAndListener>> batchedRequestsListener;
    private InferenceService inferenceService;
    private Model model;
    private final ChunkingSettings chunkingSettings;
    private final AtomicInteger resultCount = new AtomicInteger();
    private List<SemanticChunkerUtils.SemanticChunkData> chunks;
    private final String input;

    public ClusterSemanticChunker(
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
                List.of(chunk.sentence),
                false,
                null,
                null,
                TimeValue.ONE_HOUR,
                new DebatchingListener(chunk, chunks.size())
            );
        }
    }

    private void postInference() {
        calculateCosineSimilarityToNextForChunks();
        var chunksWithValidCosineSimilarityDiff = getChunksWithMaximizedCosineSimilaritySum(chunks);

        List<EmbeddingRequestChunker.BatchRequestAndListener> batchedRequests = new EmbeddingRequestChunker(
            List.of(input), // TODO: Change this to just input when fixing the input type
            List.of(chunksWithValidCosineSimilarityDiff),
            EMBEDDING_MAX_BATCH_SIZE
        ).batchRequestsWithListeners(finalListener);

        batchedRequestsListener.onResponse(batchedRequests);
    }

    private void calculateCosineSimilarityToNextForChunks() {
        for (int i = 0; i < chunks.size() - 1; i++) {
            chunks.get(i)
                .setCosineSimilarityToNext(
                    EmbeddingResultUtils.cosineSimilarity(chunks.get(i).inferenceServiceResults, chunks.get(i + 1).inferenceServiceResults)
                );
        }
    }

    private List<Chunker.ChunkOffset> getChunksWithMaximizedCosineSimilaritySum(List<SemanticChunkerUtils.SemanticChunkData> chunks) {
        List<Chunker.ChunkOffset> bestGroupingsOffsets = new ArrayList<>();
        boolean[] usedChunks = new boolean[chunks.size()];

        var maxChunkSize = chunkingSettings instanceof ClusterSemanticChunkingSettings
            ? ((ClusterSemanticChunkingSettings) chunkingSettings).getMaxChunkSize()
            : 400;

        while (true) {
            List<SemanticChunkerUtils.SemanticChunkData> bestGrouping = new ArrayList<>();
            double maxCosineSimilaritySum = 0;

            for (int i = 0; i < chunks.size(); i++) {
                if (usedChunks[i]) continue;

                List<SemanticChunkerUtils.SemanticChunkData> currentGroup = new ArrayList<>();
                currentGroup.add(chunks.get(i));
                double currentCosineSimilaritySum = 0;
                int currentLength = chunks.get(i).chunkOffset.end() - chunks.get(i).chunkOffset.start();

                for (int j = i; j < chunks.size(); j++) {
                    if (usedChunks[j]) break;

                    SemanticChunkerUtils.SemanticChunkData chunk = chunks.get(j);
                    int chunkLength = chunk.chunkOffset.end() - chunk.chunkOffset.start();

                    if (currentLength + chunkLength <= maxChunkSize) {
                        currentGroup.add(chunk);
                        currentLength += chunkLength;

                        if (currentGroup.size() > 1) {
                            currentCosineSimilaritySum += currentGroup.get(currentGroup.size() - 2).cosineSimilarityToNext;
                        }

                        if (currentCosineSimilaritySum > maxCosineSimilaritySum) {
                            maxCosineSimilaritySum = currentCosineSimilaritySum;
                            bestGrouping = new ArrayList<>(currentGroup);
                        }
                    } else {
                        break;
                    }
                }
            }

            if (bestGrouping.isEmpty()) {
                break;
            }

            bestGroupingsOffsets.add(
                new Chunker.ChunkOffset(
                    bestGrouping.get(0).chunkOffset.start(),
                    bestGrouping.get(bestGrouping.size() - 1).chunkOffset.end()
                )
            );
            for (SemanticChunkerUtils.SemanticChunkData chunk : bestGrouping) {
                usedChunks[chunks.indexOf(chunk)] = true;
            }
        }

        // Add remaining chunks in the bestGroupingsOffsets
        for (int i = 0; i < usedChunks.length; i++) {
            if (usedChunks[i] == false) {
                bestGroupingsOffsets.add(new Chunker.ChunkOffset(chunks.get(i).chunkOffset.start(), chunks.get(i).chunkOffset.end()));
            }
        }

        return bestGroupingsOffsets.stream().sorted(Comparator.comparingInt(Chunker.ChunkOffset::start)).toList();
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
