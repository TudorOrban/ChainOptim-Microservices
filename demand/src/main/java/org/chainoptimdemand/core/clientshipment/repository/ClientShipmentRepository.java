package org.chainoptimdemand.core.clientshipment.repository;

import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ClientShipmentRepository extends JpaRepository<ClientShipment, Integer>, ClientShipmentsSearchRepository {

    @Query("SELECT ss FROM ClientShipment ss " +
            "WHERE ss.clientOrderId = :orderId")
    List<ClientShipment> findBySupplyOrderId(@Param("orderId") Integer orderId);

    @Query("SELECT ss FROM ClientShipment ss " +
            "WHERE ss.clientOrderId IN :orderIds")
    List<ClientShipment> findBySupplyOrderIds(@Param("orderIds") List<Integer> orderIds);

    @Query("SELECT ss FROM ClientShipment ss " +
            "WHERE ss.clientOrderId IN :orderIds")
    List<ClientShipment> findByClientOrderIds(@Param("orderIds") List<Integer> clientOrderIds);

    @Query("SELECT COUNT(ss) FROM ClientShipment ss, ClientOrder so WHERE ss.clientOrderId = so.id AND so.organizationId = :organizationId")
    long countByOrganizationId(@Param("organizationId") Integer organizationId);
}
