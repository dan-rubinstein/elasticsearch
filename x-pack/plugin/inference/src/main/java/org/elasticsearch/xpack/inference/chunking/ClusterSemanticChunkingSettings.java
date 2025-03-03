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

public class ClusterSemanticChunkingSettings implements ChunkingSettings {
    public static final String NAME = "ClusterSemanticChunkingSettings";
    private static final ChunkingStrategy STRATEGY = ChunkingStrategy.CLUSTER_SEMANTIC;

    protected final int max_chunk_size;

    public ClusterSemanticChunkingSettings(Integer max_chunk_size) {
        this.max_chunk_size = max_chunk_size;
    }

    public static ClusterSemanticChunkingSettings fromMap(Map<String, Object> map) {
        ValidationException validationException = new ValidationException();

        Integer max_chunk_size = ServiceUtils.extractRequiredPositiveInteger(
            map,
            "max_chunk_size", // TODO: Make this a constant
            ModelConfigurations.CHUNKING_SETTINGS,
            validationException
        );

        if (validationException.validationErrors().isEmpty() == false) {
            throw validationException;
        }

        return new ClusterSemanticChunkingSettings(max_chunk_size);
    }

    public int getMaxChunkSize() {
        return max_chunk_size;
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
        out.write(max_chunk_size);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        {
            builder.field(ChunkingSettingsOptions.STRATEGY.toString(), STRATEGY);
            builder.field("max_chunk_size", max_chunk_size);
        }
        builder.endObject();
        return builder;
    }
}
