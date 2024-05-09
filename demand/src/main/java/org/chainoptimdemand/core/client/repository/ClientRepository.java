package org.chainoptimdemand.core.client.repository;

import org.chainoptimdemand.core.client.model.Client;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ClientRepository extends JpaRepository<Client, Integer>, ClientsSearchRepository {

    List<Client> findByOrganizationId(Integer organizationId);

    @Query("SELECT p.organizationId FROM Client p WHERE p.id = :clientId")
    Optional<Integer> findOrganizationIdById(@Param("clientId") Long clientId);

    @Query("SELECT p FROM Client p WHERE p.name = :name")
    Optional<Client> findByName(@Param("name") String name);

    long countByOrganizationId(Integer organizationId);
}