package org.chainoptimdemand.core.clientorder.repository;

import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ClientOrderRepository extends JpaRepository<ClientOrder, Integer>, ClientOrdersSearchRepository {

    List<ClientOrder> findByOrganizationId(Integer organizationId);

    List<ClientOrder> findByClientId(Integer clientId);

    @Query("SELECT so FROM ClientOrder so WHERE so.id IN :ids")
    Optional<List<ClientOrder>> findByIds(@Param("ids") List<Integer> ids);

    @Query("SELECT so.organizationId FROM ClientOrder so WHERE so.id = :clientOrderId")
    Optional<Integer> findOrganizationIdById(@Param("clientOrderId") Long clientOrderId);

    @Query("SELECT so FROM ClientOrder so WHERE so.companyId = :companyId")
    Optional<ClientOrder> findByCompanyId(@Param("companyId") String companyId);

    @Query("SELECT so FROM ClientOrder so WHERE so.companyId IN :companyIds")
    List<ClientOrder> findByCompanyIds(@Param("companyIds") List<String> companyIds);

    @Query("SELECT COUNT(so) FROM ClientOrder so WHERE so.organizationId = :organizationId")
    long countByOrganizationId(@Param("organizationId") Integer organizationId);

}
