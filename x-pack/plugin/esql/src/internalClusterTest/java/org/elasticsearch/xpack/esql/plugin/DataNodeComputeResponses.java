/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.plugin;

import org.elasticsearch.compute.operator.DriverCompletionInfo;
import org.elasticsearch.transport.TransportResponse;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Test-only access to {@link DataNodeComputeResponse}, which is package-private.
 *
 * <p>Lets a test make a data node report partial work, as a lenient external read that dropped records would, without needing a real
 * external source: the coordinator only ever sees the {@link DriverCompletionInfo} in the response.</p>
 */
public final class DataNodeComputeResponses {

    private DataNodeComputeResponses() {}

    /**
     * Returns {@code response} with its completion info marked partial and {@code warning} added to its warnings. A response that is
     * not a {@link DataNodeComputeResponse} is returned unchanged.
     */
    public static TransportResponse markPartial(TransportResponse response, String warning) {
        if (response instanceof DataNodeComputeResponse dataNodeResponse) {
            DriverCompletionInfo info = dataNodeResponse.completionInfo();
            Set<String> warnings = new LinkedHashSet<>(info.warnings());
            warnings.add(warning);
            return new DataNodeComputeResponse(
                new DriverCompletionInfo(
                    info.documentsFound(),
                    info.valuesLoaded(),
                    info.rowsEmitted(),
                    info.bytesRead(),
                    info.readNanos(),
                    info.readCpuNanos(),
                    info.cpuNanos(),
                    info.driverNanos(),
                    info.driverProfiles(),
                    info.planProfiles(),
                    info.capturedSourceMetadata(),
                    true,
                    info.approximationApplied(),
                    warnings
                ),
                dataNodeResponse.shardLevelFailures()
            );
        }
        return response;
    }
}
