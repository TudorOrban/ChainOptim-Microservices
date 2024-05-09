package org.chainoptimsupply.internal.in.tenant.service;

import org.chainoptimsupply.shared.kafka.OrganizationEvent;

public interface OrganizationChangesListenerService {

    void listenToOrganizationEvent(OrganizationEvent event);
}
