package org.chainoptimsupply.shared;

import lombok.AllArgsConstructor;

import java.util.List;

@AllArgsConstructor
public class PaginatedResults<T> {
    public List<T> results;
    public long totalCount;
}
