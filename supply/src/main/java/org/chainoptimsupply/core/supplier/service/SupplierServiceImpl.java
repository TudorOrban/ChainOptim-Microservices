package org.chainoptimsupply.core.supplier.service;

import org.chainoptimsupply.core.supplier.dto.CreateSupplierDTO;
import org.chainoptimsupply.core.supplier.dto.SupplierDTOMapper;
import org.chainoptimsupply.core.supplier.dto.SuppliersSearchDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierDTO;
import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.core.supplier.repository.SupplierRepository;
import org.chainoptimsupply.exception.PlanLimitReachedException;
import org.chainoptimsupply.exception.ResourceNotFoundException;
import org.chainoptimsupply.exception.ValidationException;
import org.chainoptimsupply.internal.subscriptionplan.service.SubscriptionPlanLimiterService;
import org.chainoptimsupply.shared.enums.Feature;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.internal.location.dto.Location;
import org.chainoptimsupply.internal.location.service.LocationService;
import org.chainoptimsupply.shared.sanitization.EntitySanitizerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SupplierServiceImpl implements SupplierService {

    private final SupplierRepository supplierRepository;
    private final LocationService locationService;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public SupplierServiceImpl(SupplierRepository supplierRepository,
                               LocationService locationService,
                               SubscriptionPlanLimiterService planLimiterService,
                               EntitySanitizerService entitySanitizerService) {
        this.supplierRepository = supplierRepository;
        this.locationService = locationService;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<Supplier> getAllSuppliers() {
        return supplierRepository.findAll();
    }

    public Supplier getSupplierById(Integer supplierId) {
        return supplierRepository.findById(supplierId)
                .orElseThrow(() -> new ResourceNotFoundException("Supplier with ID: " + supplierId + " not found."));
    }

    public List<Supplier> getSuppliersByOrganizationId(Integer organizationId) {
        return supplierRepository.findByOrganizationId(organizationId);
    }

    public PaginatedResults<SuppliersSearchDTO> getSuppliersByOrganizationIdAdvanced(Integer organizationId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        PaginatedResults<Supplier> paginatedResults = supplierRepository.findByOrganizationIdAdvanced(organizationId, searchQuery, sortBy, ascending, page, itemsPerPage);
        return new PaginatedResults<>(
            paginatedResults.results.stream()
            .map(SupplierDTOMapper::convertToSuppliersSearchDTO)
            .toList(),
            paginatedResults.totalCount
        );
    }

    public Supplier createSupplier(CreateSupplierDTO supplierDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(supplierDTO.getOrganizationId(), Feature.SUPPLIER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed suppliers for the current Subscription Plan.");
        }

        // Sanitize input
        CreateSupplierDTO sanitizedSupplierDTO = entitySanitizerService.sanitizeCreateSupplierDTO(supplierDTO);

        // Create location if requested
        if (sanitizedSupplierDTO.isCreateLocation() && sanitizedSupplierDTO.getLocation() != null) {
            Location location = locationService.createLocation(sanitizedSupplierDTO.getLocation());
            Supplier supplier = SupplierDTOMapper.convertCreateSupplierDTOToSupplier(sanitizedSupplierDTO);
            supplier.setLocationId(location.getId());
            return supplierRepository.save(supplier);
        } else {
            return supplierRepository.save(SupplierDTOMapper.convertCreateSupplierDTOToSupplier(sanitizedSupplierDTO));
        }
    }

    public Supplier updateSupplier(UpdateSupplierDTO supplierDTO) {
        UpdateSupplierDTO sanitizedSupplierDTO = entitySanitizerService.sanitizeUpdateSupplierDTO(supplierDTO);

        Supplier supplier = supplierRepository.findById(sanitizedSupplierDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Supplier with ID: " + sanitizedSupplierDTO.getId() + " not found."));

        supplier.setName(sanitizedSupplierDTO.getName());

        // Create new supplier or use existing or throw if not provided
        Location location;
        if (sanitizedSupplierDTO.isCreateLocation() && sanitizedSupplierDTO.getLocation() != null) {
            location = locationService.createLocation(sanitizedSupplierDTO.getLocation());
        } else if (sanitizedSupplierDTO.getLocationId() != null) {
            location = new Location();
            location.setId(sanitizedSupplierDTO.getLocationId());
        } else {
            throw new ValidationException("Location is required.");
        }
        supplier.setLocationId(location.getId());

        supplierRepository.save(supplier);
        return supplier;
    }

    public void deleteSupplier(Integer supplierId) {
        Supplier supplier = new Supplier();
        supplier.setId(supplierId);
        supplierRepository.delete(supplier);
    }
}
