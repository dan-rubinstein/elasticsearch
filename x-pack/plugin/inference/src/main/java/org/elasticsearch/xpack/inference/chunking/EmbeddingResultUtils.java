/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.chunking;

import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xpack.core.inference.results.SparseEmbeddingResults;
import org.elasticsearch.xpack.core.ml.search.WeightedToken;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class EmbeddingResultUtils {
    public static float cosineDistance(InferenceServiceResults a, InferenceServiceResults b) {
        if (a == null || b == null) {
            return 0.0f;
        }
        if (a instanceof SparseEmbeddingResults embeddingResultsA && b instanceof SparseEmbeddingResults embeddingResultsB) {
            return (1 - cosineSimilarity(embeddingResultsA, embeddingResultsB));
        } else {
            throw new ElasticsearchStatusException("InferenceServiceResults must be of type EmbeddingResults", RestStatus.BAD_REQUEST);
        }
    }

    public static float cosineSimilarity(InferenceServiceResults a, InferenceServiceResults b) {
        if (a == null || b == null) {
            return 0.0f;
        }
        if (a instanceof SparseEmbeddingResults embeddingResultsA && b instanceof SparseEmbeddingResults embeddingResultsB) {
            return cosineSimilarity(embeddingResultsA, embeddingResultsB);
        } else {
            throw new ElasticsearchStatusException("InferenceServiceResults must be of type EmbeddingResults", RestStatus.BAD_REQUEST);
        }
    }

    private static float cosineSimilarity(SparseEmbeddingResults a, SparseEmbeddingResults b) {
        var embeddingKeys = calculateEmbeddingKeys(a, b);
        var vectorA = extractEmbeddingVector(a.embeddings(), embeddingKeys);
        var vectorB = extractEmbeddingVector(b.embeddings(), embeddingKeys);

        var dotProduct = vectorA.entrySet()
            .stream()
            .mapToDouble(entry -> entry.getValue() * vectorB.getOrDefault(entry.getKey(), 0.0f))
            .sum();

        var magnitudeA = Math.sqrt(vectorA.values().stream().mapToDouble(value -> value * value).sum());
        var magnitudeB = Math.sqrt(vectorB.values().stream().mapToDouble(value -> value * value).sum());

        return (float) (1 - (dotProduct / (magnitudeA * magnitudeB)));
    }

    private static List<String> calculateEmbeddingKeys(SparseEmbeddingResults a, SparseEmbeddingResults b) {
        var embeddingKeysA = a.embeddings().stream().flatMap(embedding -> embedding.tokens().stream()).map(WeightedToken::token).toList();

        var embeddingKeysB = b.embeddings().stream().flatMap(embedding -> embedding.tokens().stream()).map(WeightedToken::token).toList();

        return Stream.concat(embeddingKeysA.stream(), embeddingKeysB.stream()).distinct().toList();
    }

    private static Map<Integer, Float> extractEmbeddingVector(
        List<SparseEmbeddingResults.Embedding> embeddings,
        List<String> embeddingKeys
    ) {
        Map<Integer, Float> embeddingVector = new HashMap<>();

        for (int i = 0; i < embeddingKeys.size(); i++) {
            var key = embeddingKeys.get(i);
            var weight = embeddings.stream()
                .flatMap(embedding -> embedding.tokens().stream())
                .filter(token -> token.token().equals(key))
                .map(WeightedToken::weight)
                .findFirst()
                .orElse(0.0f);
            embeddingVector.put(i, weight);
        }
        return embeddingVector;
    }

    public static float getPercentile(List<Float> distances, float percentile) {
        if (distances.isEmpty()) {
            return 0.0f;
        }
        var sortedDistances = distances.stream().sorted().toList();
        var index = (int) Math.ceil(percentile * (sortedDistances.size() - 1));
        return sortedDistances.get(index);
    }
}
