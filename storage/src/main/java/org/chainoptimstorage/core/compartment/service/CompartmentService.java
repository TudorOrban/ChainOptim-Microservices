package org.chainoptimstorage.core.compartment.service;

import org.chainoptimstorage.core.compartment.dto.CreateCompartmentDTO;
import org.chainoptimstorage.core.compartment.dto.UpdateCompartmentDTO;
import org.chainoptimstorage.core.compartment.model.Compartment;

import java.util.List;

public interface CompartmentService {

    List<Compartment> getCompartmentsByOrganizationId(Integer organizationId);
    List<Compartment> getCompartmentsByWarehouseId(Integer warehouseId);
    Compartment getCompartmentById(Integer compartmentId);
    Compartment createCompartment(CreateCompartmentDTO compartmentDTO);
    Compartment updateCompartment(UpdateCompartmentDTO compartmentDTO);
    void deleteCompartment(Integer compartmentId);
}
