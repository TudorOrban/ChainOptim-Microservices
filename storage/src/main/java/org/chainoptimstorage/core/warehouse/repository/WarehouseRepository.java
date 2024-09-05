package org.chainoptimstorage.core.warehouse.repository;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface WarehouseRepository extends JpaRepository<Warehouse, Integer>, WarehousesSearchRepository {

    List<Warehouse> findByOrganizationId(Integer organizationId);

    @Query("SELECT p.organizationId FROM Supplier p WHERE p.id = :supplierId")
    Optional<Integer> findOrganizationIdById(@Param("supplierId") Long supplierId);

    @Query("SELECT p FROM Supplier p WHERE p.name = :name")
    Optional<Warehouse> findByName(@Param("name") String name);

    long countByOrganizationId(Integer organizationId);
}