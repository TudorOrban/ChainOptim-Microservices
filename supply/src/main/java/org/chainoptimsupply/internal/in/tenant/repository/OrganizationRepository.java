package org.chainoptimsupply.internal.in.tenant.repository;


import org.chainoptimsupply.internal.in.tenant.model.Organization;

public interface OrganizationRepository {

    Organization getOrganizationWithUsersAndCustomRoles(Integer id);
    Organization.SubscriptionPlanTier getSubscriptionPlanTierById(Integer id);
}
