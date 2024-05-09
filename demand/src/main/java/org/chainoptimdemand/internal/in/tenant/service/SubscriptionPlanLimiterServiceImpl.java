package org.chainoptimdemand.internal.in.tenant.service;

import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.internal.in.tenant.model.PlanDetails;
import org.chainoptimdemand.internal.in.tenant.model.SubscriptionPlans;
import org.chainoptimdemand.internal.in.tenant.repository.OrganizationRepository;
import org.chainoptimdemand.shared.enums.Feature;
import org.chainoptimdemand.core.client.repository.ClientRepository;
import org.chainoptimdemand.internal.in.tenant.model.Organization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SubscriptionPlanLimiterServiceImpl implements SubscriptionPlanLimiterService {

    private final OrganizationRepository organizationRepository;
    private final ClientRepository clientRepository;

    @Autowired
    public SubscriptionPlanLimiterServiceImpl(OrganizationRepository organizationRepository,
                                              ClientRepository clientRepository) {
        this.organizationRepository = organizationRepository;
        this.clientRepository = clientRepository;
    }

    public boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity) {
        Organization.SubscriptionPlanTier planTier = organizationRepository.getSubscriptionPlanTierById(organizationId);
        if (planTier == null) throw new ResourceNotFoundException("Organization not found");
        PlanDetails planDetails = SubscriptionPlans.getPlans().get(planTier);

        if (planTier.equals(Organization.SubscriptionPlanTier.PRO)) return false; // No limits for PRO plan

        return switch (feature) {
            case Feature.SUPPLIER -> {
                long clientsCount = clientRepository.countByOrganizationId(organizationId);
                yield clientsCount + quantity >= planDetails.getMaxClients();
            }
            case Feature.SUPPLIER_ORDER -> {
//                long clientOrdersCount = clientRepository.getClientOrdersCount(organizationId);
                long clientOrdersCount = 0;
                yield clientOrdersCount + quantity >= planDetails.getMaxClientOrders();
            }
            case Feature.SUPPLIER_SHIPMENT -> {
//                long clientShipmentsCount = clientRepository.getClientShipmentsCount(organizationId);
                long clientShipmentsCount = 0;
                yield clientShipmentsCount + quantity >= planDetails.getMaxClientShipments();
            }
            default -> false;
        };
    }
}
