package org.chainoptimsupply.tenant;


import org.chainoptimsupply.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
    Organization.SubscriptionPlanTier getSubscriptionPlanTierById(Integer id);
}
