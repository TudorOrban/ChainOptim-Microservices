package org.chainoptimsupply.internal.tenant;


import org.chainoptimsupply.internal.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
    Organization.SubscriptionPlanTier getSubscriptionPlanTierById(Integer id);
}
