package org.chainoptimnotifications.shared.search.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PaginatedResults<T> {

    private List<T> results;
    private long totalCount;
}
