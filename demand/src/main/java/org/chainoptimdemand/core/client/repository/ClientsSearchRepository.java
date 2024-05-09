package org.chainoptimdemand.core.client.repository;


import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.shared.PaginatedResults;

public interface ClientsSearchRepository {
    PaginatedResults<Client> findByOrganizationIdAdvanced(
            Integer organizationId,
            String searchQuery,
            String sortBy,
            boolean ascending,
            int page,
            int itemsPerPage
    );
}
