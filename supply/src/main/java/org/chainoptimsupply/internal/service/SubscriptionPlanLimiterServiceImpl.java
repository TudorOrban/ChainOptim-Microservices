package org.chainoptimsupply.internal.service;

import org.chainoptimsupply.kafka.Feature;
import org.chainoptimsupply.tenant.OrganizationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SubscriptionPlanLimiterServiceImpl implements SubscriptionPlanLimiterService {

    private final OrganizationRepository organizationRepository;

    @Autowired
    public SubscriptionPlanLimiterServiceImpl(OrganizationRepository organizationRepository) {
        this.organizationRepository = organizationRepository;
    }

    public boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity) {
        Organization.SubscriptionPlanTier planTier = organizationRepository.getSubscriptionPlanTierById(organizationId)
                .orElseThrow(() -> new ResourceNotFoundException("Organization with ID: " + organizationId + " not found"));
        PlanDetails planDetails = SubscriptionPlans.getPlans().get(planTier);

        if (planTier.equals(Organization.SubscriptionPlanTier.PRO)) return false; // No limits for PRO plan

        return switch (feature) {
            case Feature.PRODUCT -> snapshot.getProductsCount() + quantity >= planDetails.getMaxProducts();
            case Feature.PRODUCT_STAGE -> snapshot.getProductStagesCount() + quantity >= planDetails.getMaxProductStages();
            case Feature.COMPONENT -> snapshot.getComponentsCount() + quantity >= planDetails.getMaxComponents();
            case Feature.FACTORY -> snapshot.getFactoriesCount() + quantity >= planDetails.getMaxFactories();
            case Feature.FACTORY_INVENTORY -> snapshot.getFactoryInventoryItemsCount() + quantity >= planDetails.getMaxFactoryInventoryItems();
            case Feature.FACTORY_STAGE -> snapshot.getFactoryStagesCount() + quantity >= planDetails.getMaxFactoryStages();
            case Feature.WAREHOUSE -> snapshot.getWarehousesCount() + quantity >= planDetails.getMaxWarehouses();
            case Feature.WAREHOUSE_INVENTORY -> snapshot.getWarehouseInventoryItemsCount() + quantity >= planDetails.getMaxWarehouseInventoryItems();
            case Feature.SUPPLIER -> snapshot.getSuppliersCount() + quantity >= planDetails.getMaxSuppliers();
            case Feature.SUPPLIER_ORDER -> snapshot.getSupplierOrdersCount() + quantity >= planDetails.getMaxSupplierOrders();
            case Feature.SUPPLIER_SHIPMENT -> snapshot.getSupplierShipmentsCount() + quantity >= planDetails.getMaxSupplierShipments();
            case Feature.CLIENT -> snapshot.getClientsCount() + quantity >= planDetails.getMaxClients();
            case Feature.CLIENT_ORDER -> snapshot.getClientOrdersCount() + quantity >= planDetails.getMaxClientOrders();
            case Feature.CLIENT_SHIPMENT -> snapshot.getClientShipmentsCount() + quantity >= planDetails.getMaxClientShipments();
            default -> true; // Don't restrict here for now
        };
    }
}
