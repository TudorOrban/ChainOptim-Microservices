package org.chainoptimstorage.core.compartment.service;

import org.chainoptimstorage.exception.PlanLimitReachedException;
import org.chainoptimstorage.core.compartment.dto.CompartmentDTOMapper;
import org.chainoptimstorage.core.compartment.dto.CreateCompartmentDTO;
import org.chainoptimstorage.core.compartment.dto.UpdateCompartmentDTO;
import org.chainoptimstorage.core.compartment.model.Compartment;
import org.chainoptimstorage.core.compartment.repository.CompartmentRepository;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.shared.enums.Feature;
import org.apache.kafka.common.errors.ResourceNotFoundException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CompartmentServiceImpl implements CompartmentService {

    private final CompartmentRepository compartmentRepository;
    private final SubscriptionPlanLimiterService planLimiterService;

    @Autowired
    public CompartmentServiceImpl(CompartmentRepository compartmentRepository,
                                  SubscriptionPlanLimiterService planLimiterService) {
        this.compartmentRepository = compartmentRepository;
        this.planLimiterService = planLimiterService;
    }

    public List<Compartment> getCompartmentsByOrganizationId(Integer organizationId) {
        return compartmentRepository.findByOrganizationId(organizationId);
    }

    public List<Compartment> getCompartmentsByWarehouseId(Integer warehouseId) {
        return compartmentRepository.findByWarehouseId(warehouseId);
    }

    public Compartment getCompartmentById(Integer compartmentId) {
        return compartmentRepository.findById(compartmentId)
                .orElseThrow(() -> new ResourceNotFoundException("Compartment with ID: " + compartmentId + " not found."));
    }

    public Compartment createCompartment(CreateCompartmentDTO compartmentDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(compartmentDTO.getOrganizationId(), Feature.COMPARTMENT, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed compartments for the current Subscription Plan.");
        }

        Compartment compartment = CompartmentDTOMapper.mapCreateCompartmentDTOToCompartment(compartmentDTO);
        return compartmentRepository.save(compartment);
    }

    public Compartment updateCompartment(UpdateCompartmentDTO compartmentDTO) {
        Compartment compartment = compartmentRepository.findById(compartmentDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Compartment with ID: " + compartmentDTO.getId() + " not found."));

        Compartment updatedCompartment = CompartmentDTOMapper.setUpdateCompartmentDTOToCompartment(compartmentDTO, compartment);
        return compartmentRepository.save(updatedCompartment);
    }

    public void deleteCompartment(Integer compartmentId) {
        compartmentRepository.deleteById(compartmentId);
    }
}
