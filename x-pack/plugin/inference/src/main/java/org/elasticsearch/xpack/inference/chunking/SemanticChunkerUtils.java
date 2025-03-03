/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.chunking;

import com.ibm.icu.text.BreakIterator;

import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.InferenceServiceResults;

import java.util.ArrayList;
import java.util.List;

public class SemanticChunkerUtils {

    public static List<SemanticChunkData> chunk(String input, ChunkingSettings chunkingSettings) {

        if (chunkingSettings instanceof SemanticChunkingSettings semanticChunkingSettings) {
            return chunk(input, semanticChunkingSettings.bufferSize);
        }
        if (chunkingSettings instanceof ClusterSemanticChunkingSettings clusterSemanticChunkingSettings) {
            return chunk(input);
        } else {
            throw new IllegalArgumentException(
                "SemanticChunker can't use ChunkingSettings with strategy " + chunkingSettings.getChunkingStrategy()
            );
        }
    }

    private static List<SemanticChunkData> chunk(String input) {
        var sentences = splitSentences(input);
        return sentences;
    }

    private static List<SemanticChunkData> chunk(String input, int bufferSize) {
        var sentences = splitSentences(input);

        var combinedSentences = combineSentences(sentences, bufferSize);
        return combinedSentences;
    }

    private static List<SemanticChunkData> splitSentences(String input) {
        var sentenceIterator = BreakIterator.getSentenceInstance();
        sentenceIterator.setText(input);

        int sentenceStart = 0;
        int boundary = sentenceIterator.next();
        int index = 0;
        List<SemanticChunkData> sentences = new ArrayList<>();

        while (boundary != BreakIterator.DONE) {
            int sentenceEnd = sentenceIterator.current();

            String sentence = input.substring(sentenceStart, sentenceEnd);
            sentences.add(new SemanticChunkData(sentence, new Chunker.ChunkOffset(sentenceStart, sentenceEnd), null, index, null));

            index++;
            sentenceStart = sentenceEnd;
            boundary = sentenceIterator.next();
        }
        return sentences;
    }

    private static List<SemanticChunkData> combineSentences(List<SemanticChunkData> sentences, int bufferSize) {
        for (int i = 0; i < sentences.size(); i++) {
            StringBuilder combinedSentence = new StringBuilder();
            for (int j = i - bufferSize; j < i; j++) {
                if (j < 0) {
                    continue;
                }
                combinedSentence.append(sentences.get(j).sentence);
            }
            combinedSentence.append(sentences.get(i).sentence);

            for (int j = i + 1; j <= i + bufferSize; j++) {
                if (j >= sentences.size()) {
                    break;
                }
                combinedSentence.append(sentences.get(j).sentence);
            }

            sentences.get(i).setCombinedSentence(combinedSentence.toString());
        }

        return sentences;
    }

    public static class SemanticChunkData {
        public final String sentence;
        public Chunker.ChunkOffset chunkOffset;
        public String combinedSentence;
        public final int index;
        public InferenceServiceResults inferenceServiceResults;
        public float cosineSimilarityToNext;
        public float cosineDistanceToNext;

        public SemanticChunkData(
            String sentence,
            Chunker.ChunkOffset chunkOffset,
            String combinedSentence,
            int index,
            InferenceServiceResults inferenceServiceResults
        ) {
            this.sentence = sentence;
            this.chunkOffset = chunkOffset;
            this.combinedSentence = combinedSentence;
            this.index = index;
            this.inferenceServiceResults = inferenceServiceResults;
        }

        public void setCosineSimilarityToNext(float cosineSimilarityToNext) {
            this.cosineSimilarityToNext = cosineSimilarityToNext;
        }

        public void setCosineDistanceToNext(float cosineDistanceToNext) {
            this.cosineDistanceToNext = cosineDistanceToNext;
        }

        public void setCombinedSentence(String combinedSentence) {
            this.combinedSentence = combinedSentence;
        }

        public void setInferenceServiceResults(InferenceServiceResults inferenceServiceResults) {
            this.inferenceServiceResults = inferenceServiceResults;
        }

        public void setChunkOffset(Chunker.ChunkOffset chunkOffset) {
            this.chunkOffset = chunkOffset;
        }
    }
}
