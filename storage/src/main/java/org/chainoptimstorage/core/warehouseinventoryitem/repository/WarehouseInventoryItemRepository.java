package org.chainoptimstorage.core.warehouseinventoryitem.repository;

import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface WarehouseInventoryItemRepository extends JpaRepository<WarehouseInventoryItem, Integer>, WarehouseInventoryItemsSearchRepository {

    List<WarehouseInventoryItem> findByOrganizationId(Integer organizationId);

    List<WarehouseInventoryItem> findByWarehouseId(Integer warehouseId);

    @Query("SELECT so FROM WarehouseInventoryItem so WHERE so.id IN :ids")
    Optional<List<WarehouseInventoryItem>> findByIds(@Param("ids") List<Integer> ids);

    @Query("SELECT so.organizationId FROM WarehouseInventoryItem so WHERE so.id = :warehouseInventoryItemId")
    Optional<Integer> findOrganizationIdById(@Param("warehouseInventoryItemId") Long warehouseInventoryItemId);

    @Query("SELECT so FROM WarehouseInventoryItem so WHERE so.companyId = :companyId")
    Optional<WarehouseInventoryItem> findByCompanyId(@Param("companyId") String companyId);

    @Query("SELECT so FROM WarehouseInventoryItem so WHERE so.companyId IN :companyIds")
    List<WarehouseInventoryItem> findByCompanyIds(@Param("companyIds") List<String> companyIds);

    @Query("SELECT COUNT(so) FROM WarehouseInventoryItem so WHERE so.organizationId = :organizationId")
    long countByOrganizationId(@Param("organizationId") Integer organizationId);

}
