package org.chainoptimdemand.internal.in.tenant.service;


import org.chainoptimdemand.shared.enums.Feature;

public interface SubscriptionPlanLimiterService {

    boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity);
}
