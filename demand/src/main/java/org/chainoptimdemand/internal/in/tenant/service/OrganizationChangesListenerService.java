package org.chainoptimdemand.internal.in.tenant.service;

import org.chainoptimdemand.shared.kafka.OrganizationEvent;

public interface OrganizationChangesListenerService {

    void listenToOrganizationEvent(OrganizationEvent event);
}
