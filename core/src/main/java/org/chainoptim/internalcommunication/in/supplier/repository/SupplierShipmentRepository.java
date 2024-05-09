package org.chainoptim.internalcommunication.in.supplier.repository;

import org.chainoptim.internalcommunication.in.supplier.model.SupplierShipment;

import java.util.List;

public interface SupplierShipmentRepository {

    List<SupplierShipment> findSupplierShipmentsBySupplierOrderIds(List<Integer> orderIds);

    long countByOrganizationId(Integer organizationId);
}
