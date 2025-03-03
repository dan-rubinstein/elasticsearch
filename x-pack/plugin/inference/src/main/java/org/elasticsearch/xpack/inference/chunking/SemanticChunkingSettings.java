/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.chunking;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.inference.ChunkingSettings;
import org.elasticsearch.inference.ChunkingStrategy;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.ServiceUtils;

import java.io.IOException;
import java.util.Map;

public class SemanticChunkingSettings implements ChunkingSettings {
    public static final String NAME = "SemanticChunkingSettings";
    private static final ChunkingStrategy STRATEGY = ChunkingStrategy.SEMANTIC;

    protected final int bufferSize;
    protected final float threshold;

    public SemanticChunkingSettings(Integer bufferSize, Float threshold) {
        this.bufferSize = bufferSize;
        this.threshold = threshold;
    }

    public static SemanticChunkingSettings fromMap(Map<String, Object> map) {
        ValidationException validationException = new ValidationException();

        Integer bufferSize = ServiceUtils.extractRequiredPositiveIntegerBetween(
            map,
            "buffer_size", // TODO: Make this a constant
            0,
            10,
            ModelConfigurations.CHUNKING_SETTINGS,
            validationException
        );

        Float threshold = ServiceUtils.extractRequiredFloatBetween(
            map,
            "threshold", // TODO: Make this a constant
            0.0f,
            1.0f,
            ModelConfigurations.CHUNKING_SETTINGS,
            validationException
        );

        if (validationException.validationErrors().isEmpty() == false) {
            throw validationException;
        }

        return new SemanticChunkingSettings(bufferSize, threshold);
    }

    @Override
    public ChunkingStrategy getChunkingStrategy() {
        return STRATEGY;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return null;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.write(bufferSize);
        out.writeFloat(threshold);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        {
            builder.field(ChunkingSettingsOptions.STRATEGY.toString(), STRATEGY);
            builder.field("buffer_size", bufferSize);
            builder.field("threshold", threshold);
        }
        builder.endObject();
        return builder;
    }
}
