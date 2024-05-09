package org.chainoptimdemand.core.clientorder.service;

import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.shared.enums.SearchMode;
import org.chainoptimdemand.shared.search.SearchParams;
import org.chainoptimdemand.core.client.dto.CreateClientOrderDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientOrderDTO;

import java.util.List;

public interface ClientOrderService {

    List<ClientOrder> getClientOrdersByOrganizationId(Integer organizationId);
    List<ClientOrder> getClientOrdersByClientId(Integer clientId);
    PaginatedResults<ClientOrder> getClientOrdersAdvanced(SearchMode searchMode, Integer entity, SearchParams searchParams);
    Integer getOrganizationIdById(Long clientOrderId);
    long countByOrganizationId(Integer organizationId);

    ClientOrder createClientOrder(CreateClientOrderDTO order);
    List<ClientOrder> createClientOrdersInBulk(List<CreateClientOrderDTO> orderDTOs);
    List<ClientOrder> updateClientsOrdersInBulk(List<UpdateClientOrderDTO> orderDTOs);
    List<Integer> deleteClientOrdersInBulk(List<Integer> orders);
}
