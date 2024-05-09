package org.chainoptimdemand.clientorder.repository;

import org.chainoptimdemand.core.clientorder.repository.ClientOrderRepository;
import org.chainoptimdemand.internal.in.location.dto.Location;
import org.chainoptimdemand.core.client.model.Client;
import org.chainoptimdemand.core.clientorder.model.OrderStatus;
import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace.NONE;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@AutoConfigureTestDatabase(replace = NONE)
class ClientOrdersRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private ClientOrderRepository clientOrderRepository;

    Integer organizationId = 1;
    Integer locationId = 1;
    Integer clientId;
    Integer clientOrderId;
    Integer componentId = 1;
    @BeforeEach
    void setUp() {
        // Set up client for update and delete tests
        Client client = addTestClient();
        clientId = client.getId();

        ClientOrder clientOrder = addTestClientOrder();
        clientOrderId = clientOrder.getId();
    }

    @Test
    void testCreateClientOrder() {
        // Arrange
        ClientOrder clientOrder = addTestClientOrder();

        // Act
        ClientOrder savedClientOrder = entityManager.persist(clientOrder);
        entityManager.flush();

        // Assert
        Optional<ClientOrder> foundClientOrderOpt = clientOrderRepository.findById(savedClientOrder.getId());
        assertTrue(foundClientOrderOpt.isPresent(), "Client should be found in the database");

        ClientOrder foundClientOrder = foundClientOrderOpt.get();
        assertEquals(savedClientOrder.getOrganizationId(), foundClientOrder.getOrganizationId());
        assertEquals(savedClientOrder.getClientId(), foundClientOrder.getClientId());
        assertEquals(savedClientOrder.getOrderDate(), foundClientOrder.getOrderDate());
        assertEquals(savedClientOrder.getQuantity(), foundClientOrder.getQuantity());
    }

    @Test
    void testUpdateClient() {
        // Arrange
        Optional<ClientOrder> clientOrderOptional = clientOrderRepository.findById(clientOrderId); // Id from setUp
        if (clientOrderOptional.isEmpty()) {
            fail("Expected an existing client order with id " + clientOrderId);
        }

        ClientOrder clientOrder = clientOrderOptional.get();
        clientOrder.setStatus(OrderStatus.DELIVERED);
        clientOrder.setQuantity(20f);
        LocalDateTime deliveryDate = LocalDateTime.parse("2021-01-02T00:00:00");
        clientOrder.setDeliveryDate(deliveryDate);

        // Act
        ClientOrder updatedClientOrder = clientOrderRepository.save(clientOrder);

        // Assert
        assertNotNull(updatedClientOrder);
        assertEquals(clientOrder.getOrganizationId(), updatedClientOrder.getOrganizationId());
        assertEquals(clientOrder.getClientId(), updatedClientOrder.getClientId());
        assertEquals(clientOrder.getOrderDate(), updatedClientOrder.getOrderDate());
        assertEquals(clientOrder.getQuantity(), updatedClientOrder.getQuantity());
        assertEquals(clientOrder.getStatus(), updatedClientOrder.getStatus());
        assertEquals(clientOrder.getDeliveryDate(), updatedClientOrder.getDeliveryDate());
    }

    @Test
    void testDeleteClient() {
        // Arrange
        Optional<ClientOrder> clientOrderToBeDeletedOptional = clientOrderRepository.findById(clientOrderId);
        if (clientOrderToBeDeletedOptional.isEmpty()) {
            fail("Expected an existing client order with id " + clientOrderId);
        }

        ClientOrder clientOrderToBeDeleted = clientOrderToBeDeletedOptional.get();

        // Act
        clientOrderRepository.delete(clientOrderToBeDeleted);

        // Assert
        Optional<ClientOrder> deletedClientOrderOptional = clientOrderRepository.findById(clientOrderId);
        if (deletedClientOrderOptional.isPresent()) {
            fail("Expected client order with id 1 to have been deleted");
        }
    }

    Client addTestClient() {
        Client client = new Client();
        client.setName("Test Client");
        client.setOrganizationId(organizationId);
        Location location = new Location();
        location.setId(locationId);
        client.setLocationId(location.getId());

        return entityManager.persist(client);
    }

    ClientOrder addTestClientOrder() {
        ClientOrder clientOrder = new ClientOrder();
        clientOrder.setClientId(clientId);
        clientOrder.setOrganizationId(organizationId);
        LocalDateTime orderDate = LocalDateTime.parse("2021-01-01T00:00:00");
        clientOrder.setOrderDate(orderDate);
        clientOrder.setStatus(OrderStatus.PLACED);
        clientOrder.setQuantity(10f);
        clientOrder.setComponentId(componentId);

        return entityManager.persist(clientOrder);
    }
}
