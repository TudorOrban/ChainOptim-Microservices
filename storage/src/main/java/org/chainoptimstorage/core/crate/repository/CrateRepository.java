package org.chainoptimstorage.core.crate.repository;

import org.chainoptimstorage.core.crate.model.Crate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface CrateRepository extends JpaRepository<Crate, Integer> {

    List<Crate> findByOrganizationId(Integer organizationId);

    @Query("SELECT c.organizationId FROM Crate c WHERE c.id = :crateId")
    Optional<Integer> findOrganizationIdById(@Param("crateId") Long crateId);

    @Query("SELECT COUNT(c) FROM Crate c WHERE c.organizationId = :organizationId")
    long countByOrganizationId(@Param("organizationId") Integer organizationId);


}
