package org.chainoptimsupply.internal.in.tenant.service;


import org.chainoptimsupply.shared.enums.Feature;

public interface SubscriptionPlanLimiterService {

    boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity);
}
