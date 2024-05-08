package org.chainoptimsupply.internal.component.repository;


import org.chainoptimsupply.internal.subscriptionplan.model.Component;
import org.chainoptimsupply.tenant.model.Organization;

import java.util.Optional;

public interface ComponentRepository {

    Optional<Component> findById(Integer id);
}
