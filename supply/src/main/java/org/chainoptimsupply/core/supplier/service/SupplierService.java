package org.chainoptimsupply.core.supplier.service;


import org.chainoptimsupply.core.supplier.dto.CreateSupplierDTO;
import org.chainoptimsupply.core.supplier.dto.SuppliersSearchDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierDTO;
import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.shared.PaginatedResults;

import java.util.List;

public interface SupplierService {
    // Fetch
    List<Supplier> getAllSuppliers();
    Supplier getSupplierById(Integer id);
    List<Supplier> getSuppliersByOrganizationId(Integer organizationId);
    PaginatedResults<SuppliersSearchDTO> getSuppliersByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);

    // Create
    Supplier createSupplier(CreateSupplierDTO supplierDTO);

    // Update
    Supplier updateSupplier(UpdateSupplierDTO updateSupplierDTO);

    // Delete
    void deleteSupplier(Integer supplierId);
}
