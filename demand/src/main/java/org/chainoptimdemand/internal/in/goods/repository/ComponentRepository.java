package org.chainoptimdemand.internal.in.goods.repository;


import org.chainoptimdemand.internal.in.goods.model.Component;

import java.util.Optional;

public interface ComponentRepository {

    Optional<Component> findById(Integer id);
}
