package org.chainoptimstorage.internal.in.tenant.service;


import org.chainoptimstorage.shared.enums.Feature;

public interface SubscriptionPlanLimiterService {

    boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity);
}
