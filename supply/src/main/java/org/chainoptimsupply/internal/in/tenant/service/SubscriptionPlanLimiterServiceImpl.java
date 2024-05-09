package org.chainoptimsupply.internal.in.tenant.service;

import org.chainoptimsupply.exception.ResourceNotFoundException;
import org.chainoptimsupply.internal.in.tenant.model.PlanDetails;
import org.chainoptimsupply.internal.in.tenant.model.SubscriptionPlans;
import org.chainoptimsupply.internal.in.tenant.repository.OrganizationRepository;
import org.chainoptimsupply.shared.enums.Feature;
import org.chainoptimsupply.core.supplier.repository.SupplierRepository;
import org.chainoptimsupply.internal.in.tenant.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SubscriptionPlanLimiterServiceImpl implements SubscriptionPlanLimiterService {

    private final OrganizationRepository organizationRepository;
    private final SupplierRepository supplierRepository;

    @Autowired
    public SubscriptionPlanLimiterServiceImpl(OrganizationRepository organizationRepository,
                                              SupplierRepository supplierRepository) {
        this.organizationRepository = organizationRepository;
        this.supplierRepository = supplierRepository;
    }

    public boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity) {
        Organization.SubscriptionPlanTier planTier = organizationRepository.getSubscriptionPlanTierById(organizationId);
        if (planTier == null) throw new ResourceNotFoundException("Organization not found");
        PlanDetails planDetails = SubscriptionPlans.getPlans().get(planTier);

        if (planTier.equals(Organization.SubscriptionPlanTier.PRO)) return false; // No limits for PRO plan

        return switch (feature) {
            case Feature.SUPPLIER -> {
                long suppliersCount = supplierRepository.countByOrganizationId(organizationId);
                yield suppliersCount + quantity >= planDetails.getMaxSuppliers();
            }
            case Feature.SUPPLIER_ORDER -> {
//                long supplierOrdersCount = supplierRepository.getSupplierOrdersCount(organizationId);
                long supplierOrdersCount = 0;
                yield supplierOrdersCount + quantity >= planDetails.getMaxSupplierOrders();
            }
            case Feature.SUPPLIER_SHIPMENT -> {
//                long supplierShipmentsCount = supplierRepository.getSupplierShipmentsCount(organizationId);
                long supplierShipmentsCount = 0;
                yield supplierShipmentsCount + quantity >= planDetails.getMaxSupplierShipments();
            }
            default -> false;
        };
    }
}
