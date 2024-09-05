package org.chainoptimstorage.core.warehouse.service;

import org.chainoptimstorage.core.warehouse.dto.CreateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.dto.WarehouseDTOMapper;
import org.chainoptimstorage.core.warehouse.dto.WarehousesSearchDTO;
import org.chainoptimstorage.core.warehouse.dto.UpdateWarehouseDTO;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.exception.PlanLimitReachedException;
import org.chainoptimstorage.exception.ResourceNotFoundException;
import org.chainoptimstorage.exception.ValidationException;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.shared.enums.Feature;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.internal.in.location.dto.Location;
import org.chainoptimstorage.internal.in.location.service.LocationService;
import org.chainoptimstorage.shared.sanitization.EntitySanitizerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class WarehouseServiceImpl implements WarehouseService {

    private final WarehouseRepository warehouseRepository;
    private final LocationService locationService;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public WarehouseServiceImpl(WarehouseRepository warehouseRepository,
                                LocationService locationService,
                                SubscriptionPlanLimiterService planLimiterService,
                                EntitySanitizerService entitySanitizerService) {
        this.warehouseRepository = warehouseRepository;
        this.locationService = locationService;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<Warehouse> getAllWarehouses() {
        return warehouseRepository.findAll();
    }

    public Warehouse getWarehouseById(Integer warehouseId) {
        return warehouseRepository.findById(warehouseId)
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse with ID: " + warehouseId + " not found."));
    }

    public List<Warehouse> getWarehousesByOrganizationId(Integer organizationId) {
        return warehouseRepository.findByOrganizationId(organizationId);
    }

    public PaginatedResults<WarehousesSearchDTO> getWarehousesByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        PaginatedResults<Warehouse> paginatedResults = warehouseRepository.findByOrganizationIdAdvanced(organizationId, searchQuery, sortBy, ascending, page, itemsPerPage);
        return new PaginatedResults<>(
            paginatedResults.results.stream()
            .map(WarehouseDTOMapper::convertToWarehousesSearchDTO)
            .toList(),
            paginatedResults.totalCount
        );
    }

    public Integer getOrganizationIdById(Long warehouseId) {
        return warehouseRepository.findOrganizationIdById(warehouseId)
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse with ID: " + warehouseId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return warehouseRepository.countByOrganizationId(organizationId);
    }

    // Create
    public Warehouse createWarehouse(CreateWarehouseDTO warehouseDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(warehouseDTO.getOrganizationId(), Feature.SUPPLIER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed warehouses for the current Subscription Plan.");
        }

        // Sanitize input
        CreateWarehouseDTO sanitizedWarehouseDTO = entitySanitizerService.sanitizeCreateWarehouseDTO(warehouseDTO);

        // Create location if requested
        if (sanitizedWarehouseDTO.isCreateLocation() && sanitizedWarehouseDTO.getLocation() != null) {
            Location location = locationService.createLocation(sanitizedWarehouseDTO.getLocation());
            Warehouse warehouse = WarehouseDTOMapper.mapCreateWarehouseDTOToWarehouse(sanitizedWarehouseDTO);
            warehouse.setLocationId(location.getId());
            return warehouseRepository.save(warehouse);
        } else {
            return warehouseRepository.save(WarehouseDTOMapper.mapCreateWarehouseDTOToWarehouse(sanitizedWarehouseDTO));
        }
    }

    public Warehouse updateWarehouse(UpdateWarehouseDTO warehouseDTO) {
        UpdateWarehouseDTO sanitizedWarehouseDTO = entitySanitizerService.sanitizeUpdateWarehouseDTO(warehouseDTO);

        Warehouse warehouse = warehouseRepository.findById(sanitizedWarehouseDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse with ID: " + sanitizedWarehouseDTO.getId() + " not found."));

        warehouse.setName(sanitizedWarehouseDTO.getName());

        // Create new warehouse or use existing or throw if not provided
        Location location;
        if (sanitizedWarehouseDTO.isCreateLocation() && sanitizedWarehouseDTO.getLocation() != null) {
            location = locationService.createLocation(sanitizedWarehouseDTO.getLocation());
        } else if (sanitizedWarehouseDTO.getLocationId() != null) {
            location = new Location();
            location.setId(sanitizedWarehouseDTO.getLocationId());
        } else {
            throw new ValidationException("Location is required.");
        }
        warehouse.setLocationId(location.getId());

        warehouseRepository.save(warehouse);
        return warehouse;
    }

    public void deleteWarehouse(Integer warehouseId) {
        Warehouse warehouse = new Warehouse();
        warehouse.setId(warehouseId);
        warehouseRepository.delete(warehouse);
    }
}
