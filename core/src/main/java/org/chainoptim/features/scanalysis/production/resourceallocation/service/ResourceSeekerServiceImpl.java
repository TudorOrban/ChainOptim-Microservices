package org.chainoptim.features.scanalysis.production.resourceallocation.service;

import org.chainoptim.features.scanalysis.production.resourceallocation.model.DeficitResolution;
import org.chainoptim.features.scanalysis.production.resourceallocation.model.DeficitResolverPlan;
import org.chainoptim.features.scanalysis.production.resourceallocation.model.ResourceAllocation;
import org.chainoptim.internalcommunication.in.storage.model.Warehouse;
import org.chainoptim.internalcommunication.in.storage.model.WarehouseInventoryItem;
import org.chainoptim.internalcommunication.in.storage.repository.WarehouseInventoryItemRepository;
import org.chainoptim.internalcommunication.in.storage.repository.WarehouseRepository;
import org.chainoptim.internalcommunication.in.supplier.model.SupplierOrder;
import org.chainoptim.internalcommunication.in.supplier.model.SupplierShipment;
import org.chainoptim.internalcommunication.in.supplier.repository.SupplierOrderRepository;
import org.chainoptim.internalcommunication.in.supplier.repository.SupplierShipmentRepository;
import org.chainoptim.shared.commonfeatures.location.model.Location;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class ResourceSeekerServiceImpl implements ResourceSeekerService {

    private final WarehouseRepository warehouseRepository;
    private final WarehouseInventoryItemRepository warehouseInventoryItemRepository;
    private final SupplierOrderRepository supplierOrderRepository;
    private final SupplierShipmentRepository supplierShipmentRepository;

    @Autowired
    public ResourceSeekerServiceImpl(
            WarehouseRepository warehouseRepository,
            WarehouseInventoryItemRepository warehouseInventoryItemRepository,
            SupplierOrderRepository supplierOrderRepository,
            SupplierShipmentRepository supplierShipmentRepository) {
        this.warehouseRepository = warehouseRepository;
        this.warehouseInventoryItemRepository = warehouseInventoryItemRepository;
        this.supplierOrderRepository = supplierOrderRepository;
        this.supplierShipmentRepository = supplierShipmentRepository;
    }

    public DeficitResolverPlan seekResources(Integer organizationId, List<ResourceAllocation> allocationDeficits, Location factoryLocation) {
        DeficitResolverPlan deficitResolverPlan = new DeficitResolverPlan();
        deficitResolverPlan.setAllocationDeficits(allocationDeficits);
        List<DeficitResolution> resolutions = new ArrayList<>();

        seekWarehouseResources(organizationId, allocationDeficits, factoryLocation, resolutions);
        seekSupplyOrderResources(organizationId, allocationDeficits, factoryLocation, resolutions);

        deficitResolverPlan.setResolutions(resolutions);
        return deficitResolverPlan;
    }

    public void seekWarehouseResources(Integer organizationId,
                                        List<ResourceAllocation> allocationDeficits,
                                        Location factoryLocation,
                                        List<DeficitResolution> resolutions) {
        List<Warehouse> warehouses = warehouseRepository.findWarehousesByOrganizationId(organizationId);

        for (Warehouse warehouse : warehouses) {
            // Skip if too far
//            if (!areCloseEnough(warehouse.getLocationId(), factoryLocation, 40.0f)) continue;

            // Get warehouse inventory
            List<WarehouseInventoryItem> inventory = warehouseInventoryItemRepository.findWarehouseInventoryItemsByOrganizationId(warehouse.getId()); // TODO: Fix this

            for (ResourceAllocation resourceAllocation : allocationDeficits) {
                // Determine resource availability
                WarehouseInventoryItem correspondingItem = inventory.stream()
                        .filter(ii -> Objects.equals(ii.getComponent().getId(), resourceAllocation.getComponentId())).findAny().orElse(null);
                if (correspondingItem == null) continue;

                System.out.println("Corresponding item " + correspondingItem);
                DeficitResolution resolution = getDeficitResolution(warehouse, resourceAllocation, correspondingItem);

                resolutions.add(resolution);
            }
        }
    }

    private static DeficitResolution getDeficitResolution(Warehouse warehouse, ResourceAllocation resourceAllocation, WarehouseInventoryItem correspondingItem) {
        Float inventoryComponentQuantity = correspondingItem.getQuantity();
        Float neededComponentQuantity = resourceAllocation.getRequestedAmount() - resourceAllocation.getAllocatedAmount();
        float surplusQuantity = inventoryComponentQuantity - neededComponentQuantity;

        Float suppliedQuantity = surplusQuantity > 0 ? neededComponentQuantity : inventoryComponentQuantity;

        // Add deficit resolution
        DeficitResolution resolution = new DeficitResolution();
        resolution.setWarehouseId(warehouse.getId());
        resolution.setNeededComponentId(resourceAllocation.getComponentId());
        resolution.setNeededQuantity(neededComponentQuantity);
        resolution.setSuppliedQuantity(suppliedQuantity);
        return resolution;
    }

    public void seekSupplyOrderResources(Integer organizationId,
                                         List<ResourceAllocation> allocationDeficits,
                                         Location factoryLocation,
                                         List<DeficitResolution> resolutions) {
        List<SupplierOrder> supplierOrders = supplierOrderRepository.findSupplierOrdersByOrganizationId(organizationId);

        for (ResourceAllocation resourceAllocation : allocationDeficits) {
            // Find potential resolvers of current needed component
            List<SupplierOrder> potentialSupplierOrders = new ArrayList<>(supplierOrders.stream()
                    .filter(so -> Objects.equals(so.getComponent().getId(), resourceAllocation.getComponentId())).toList());

            // Sort by quantity to find the least amount of supplier order solutions
            potentialSupplierOrders.sort((so1, so2) -> Float.compare(so2.getQuantity(), so1.getQuantity()));

            float neededComponentQuantity = resourceAllocation.getRequestedAmount() - resourceAllocation.getAllocatedAmount();

            List<SupplierShipment> potentialShipments = supplierShipmentRepository.findSupplierShipmentsBySupplierOrderIds(
                    potentialSupplierOrders.stream().map(SupplierOrder::getId).toList()
            );

            for (SupplierOrder potentialOrder : potentialSupplierOrders) {
                List<SupplierShipment> orderShipments = potentialShipments.stream()
                        .filter(ss -> Objects.equals(ss.getSupplierOrderId(), potentialOrder.getId())).toList();

                for (SupplierShipment shipment : orderShipments) {
                    if (!areCloseEnough(shipment.getDestinationLocation(), factoryLocation, 80.0f)) continue;

                    float potentialQuantity = shipment.getQuantity();
                    float surplusQuantity = potentialQuantity - neededComponentQuantity;

                    float suppliedQuantity = surplusQuantity > 0 ? neededComponentQuantity : potentialQuantity;

                    // Set up resolution
                    DeficitResolution resolution = new DeficitResolution();
                    resolution.setNeededComponentId(resourceAllocation.getComponentId());
                    resolution.setSupplierOrderId(potentialOrder.getId());
                    resolution.setOrderShipmentId(shipment.getId());
                    resolution.setNeededQuantity(neededComponentQuantity);
                    resolution.setSuppliedQuantity(suppliedQuantity);

                    resolutions.add(resolution);

                    // Update needed quantity accordingly
//                    neededComponentQuantity -= suppliedQuantity;
                }
            }
        }
    }

    private boolean areCloseEnough(Location location1, Location location2, Float distance) {
        if (location1 == null || location2 == null) return true; // Prevent skipping if location not set
        return location1.getLatitude() - location2.getLatitude() < distance &&
                location1.getLongitude() - location2.getLongitude() < distance;
    }
}
