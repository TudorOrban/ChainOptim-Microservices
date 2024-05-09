package org.chainoptimdemand.core.clientshipment.repository;


import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;

public interface ClientShipmentsSearchRepository {
    PaginatedResults<ClientShipment> findByClientOrderIdAdvanced(
            Integer clientOrderId,
            String searchQuery,
            String sortBy,
            boolean ascending,
            int page,
            int itemsPerPage
    );
}
