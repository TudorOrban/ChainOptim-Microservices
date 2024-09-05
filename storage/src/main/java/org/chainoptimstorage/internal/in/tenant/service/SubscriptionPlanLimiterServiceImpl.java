package org.chainoptimstorage.internal.in.tenant.service;

import org.chainoptimstorage.exception.ResourceNotFoundException;
import org.chainoptimstorage.internal.in.tenant.model.PlanDetails;
import org.chainoptimstorage.internal.in.tenant.model.SubscriptionPlans;
import org.chainoptimstorage.internal.in.tenant.repository.OrganizationRepository;
import org.chainoptimstorage.shared.enums.Feature;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.internal.in.tenant.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SubscriptionPlanLimiterServiceImpl implements SubscriptionPlanLimiterService {

    private final OrganizationRepository organizationRepository;
    private final WarehouseRepository warehouseRepository;

    @Autowired
    public SubscriptionPlanLimiterServiceImpl(OrganizationRepository organizationRepository,
                                              WarehouseRepository warehouseRepository) {
        this.organizationRepository = organizationRepository;
        this.warehouseRepository = warehouseRepository;
    }

    public boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity) {
        Organization.SubscriptionPlanTier planTier = organizationRepository.getSubscriptionPlanTierById(organizationId);
        if (planTier == null) throw new ResourceNotFoundException("Organization not found");
        PlanDetails planDetails = SubscriptionPlans.getPlans().get(planTier);

        if (planTier.equals(Organization.SubscriptionPlanTier.PRO)) return false; // No limits for PRO plan

        return switch (feature) {
            case Feature.SUPPLIER -> {
                long warehousesCount = warehouseRepository.countByOrganizationId(organizationId);
                yield warehousesCount + quantity >= planDetails.getMaxWarehouses();
            }
            case Feature.WAREHOUSE_INVENTORY -> {
//                long warehouseInventoryItemsCount = warehouseRepository.getWarehouseInventoryItemsCount(organizationId);
                long warehouseInventoryItemsCount = 0;
                yield warehouseInventoryItemsCount + quantity >= planDetails.getMaxWarehouseInventoryItems();
            }
            default -> false;
        };
    }
}
