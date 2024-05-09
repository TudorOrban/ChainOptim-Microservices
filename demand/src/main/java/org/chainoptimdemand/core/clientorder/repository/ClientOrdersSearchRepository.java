package org.chainoptimdemand.core.clientorder.repository;

import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.shared.enums.SearchMode;
import org.chainoptimdemand.shared.search.SearchParams;

public interface ClientOrdersSearchRepository {
    PaginatedResults<ClientOrder> findByClientIdAdvanced(
            SearchMode searchMode,
            Integer clientId,
            SearchParams searchParams
    );
}
