package org.chainoptimstorage.warehouse.service;

import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.service.WarehouseServiceImpl;
import org.chainoptimstorage.exception.ResourceNotFoundException;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.internal.in.location.dto.CreateLocationDTO;
import org.chainoptimstorage.shared.sanitization.EntitySanitizerService;
import org.chainoptimstorage.core.warehouse.dto.CreateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.dto.WarehouseDTOMapper;
import org.chainoptimstorage.core.warehouse.dto.UpdateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class WarehouseServiceTest {

    @Mock
    private WarehouseRepository warehouseRepository;
    @Mock
    private SubscriptionPlanLimiterService planLimiterService;
    @Mock
    private EntitySanitizerService entitySanitizerService;

    @InjectMocks
    private WarehouseServiceImpl supplierService;

    @Test
    void testCreateSupplier() {
        // Arrange
        CreateWarehouseDTO supplierDTO = new CreateWarehouseDTO("Test Supplier", 1, 1, new CreateLocationDTO(), false);
        Warehouse expectedWarehouse = WarehouseDTOMapper.convertCreateSupplierDTOToSupplier(supplierDTO);

        when(warehouseRepository.save(any(Warehouse.class))).thenReturn(expectedWarehouse);
        when(planLimiterService.isLimitReached(any(), any(), any())).thenReturn(false);
        when(entitySanitizerService.sanitizeCreateSupplierDTO(any(CreateWarehouseDTO.class))).thenReturn(supplierDTO);

        // Act
        Warehouse createdWarehouse = supplierService.createSupplier(supplierDTO);

        // Assert
        assertNotNull(createdWarehouse);
        assertEquals(expectedWarehouse.getName(), createdWarehouse.getName());
        assertEquals(expectedWarehouse.getOrganizationId(), createdWarehouse.getOrganizationId());
        assertEquals(expectedWarehouse.getLocationId(), createdWarehouse.getLocationId());

        verify(warehouseRepository, times(1)).save(any(Warehouse.class));
    }

    @Test
    void testUpdateSupplier_ExistingSupplier() {
        // Arrange
        UpdateWarehouseDTO supplierDTO = new UpdateWarehouseDTO(1, "Test Supplier", 1, new CreateLocationDTO(), false);
        Warehouse existingWarehouse = new Warehouse();
        existingWarehouse.setId(1);

        when(warehouseRepository.findById(1)).thenReturn(Optional.of(existingWarehouse));
        when(warehouseRepository.save(any(Warehouse.class))).thenReturn(existingWarehouse);
        when(entitySanitizerService.sanitizeUpdateSupplierDTO(any(UpdateWarehouseDTO.class))).thenReturn(supplierDTO);

        // Act
        Warehouse updatedWarehouse = supplierService.updateSupplier(supplierDTO);

        // Assert
        assertNotNull(updatedWarehouse);
        assertEquals(existingWarehouse.getName(), updatedWarehouse.getName());
        assertEquals(existingWarehouse.getOrganizationId(), updatedWarehouse.getOrganizationId());
        assertEquals(existingWarehouse.getLocationId(), updatedWarehouse.getLocationId());

        verify(warehouseRepository, times(1)).findById(1);
    }

    @Test
    void testUpdateSupplier_NonExistingSupplier() {
        // Arrange
        UpdateWarehouseDTO supplierDTO = new UpdateWarehouseDTO(1, "Test Supplier", 1, new CreateLocationDTO(), false);
        Warehouse existingWarehouse = new Warehouse();
        existingWarehouse.setId(1);

        when(warehouseRepository.findById(1)).thenReturn(Optional.empty());
        when(entitySanitizerService.sanitizeUpdateSupplierDTO(any(UpdateWarehouseDTO.class))).thenReturn(supplierDTO);

        // Act and assert
        assertThrows(ResourceNotFoundException.class, () -> supplierService.updateSupplier(supplierDTO));

        verify(warehouseRepository, times(1)).findById(1);
        verify(warehouseRepository, never()).save(any(Warehouse.class));
    }

    @Test
    void testDeleteSupplier() {
        // Arrange
        doNothing().when(warehouseRepository).delete(any(Warehouse.class));

        // Act
        supplierService.deleteSupplier(1);

        // Assert
        verify(warehouseRepository, times(1)).delete(any(Warehouse.class));
    }
}
