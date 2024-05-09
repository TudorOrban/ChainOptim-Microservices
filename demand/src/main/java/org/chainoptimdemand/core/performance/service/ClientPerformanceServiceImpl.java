package org.chainoptimdemand.core.performance.service;

import org.chainoptimdemand.core.performance.model.ComponentDeliveryPerformance;
import org.chainoptimdemand.core.performance.model.ClientPerformanceReport;
import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.core.clientorder.repository.ClientOrderRepository;
import org.chainoptimdemand.core.clientshipment.model.ClientShipment;
import org.chainoptimdemand.core.clientshipment.repository.ClientShipmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class ClientPerformanceServiceImpl implements ClientPerformanceService {

    private final ClientOrderRepository clientOrderRepository;
    private final ClientShipmentRepository clientShipmentRepository;

    @Autowired
    public ClientPerformanceServiceImpl(ClientOrderRepository clientOrderRepository,
                                          ClientShipmentRepository clientShipmentRepository) {
        this.clientOrderRepository = clientOrderRepository;
        this.clientShipmentRepository = clientShipmentRepository;
    }

    public ClientPerformanceReport computeClientPerformanceReport(Integer clientId) {
        List<ClientOrder> clientOrders = clientOrderRepository.findByClientId(clientId);
        List<ClientShipment> clientShipments = clientShipmentRepository.findByClientOrderIds(
                clientOrders.stream().map(ClientOrder::getId).toList());

        ClientPerformanceReport report = new ClientPerformanceReport();
        Map<Integer, ComponentDeliveryPerformance> componentPerformances = new HashMap<>();

        int totalDeliveredOrders = 0;
        float totalDelays = 0;
        float ratioOfOnTimeOrderDeliveries = 0;
        float ratioOfOnTimeShipmentDeliveries = 0;
        float averageShipmentsPerOrder = 0;
        float averageTimeToShipOrder = 0;

        // Group orders by component
        Map<Integer, List<ClientOrder>> ordersByComponent = new HashMap<>();
        for (ClientOrder clientOrder : clientOrders) {
            Integer componentId = clientOrder.getComponentId();
            if (ordersByComponent.containsKey(componentId)) {
                ordersByComponent.get(componentId).add(clientOrder);
            } else {
                List<ClientOrder> orders = new ArrayList<>();
                orders.add(clientOrder);
                ordersByComponent.put(componentId, orders);
            }
        }

        for (Map.Entry<Integer, List<ClientOrder>> entry : ordersByComponent.entrySet()) {
            Integer componentId = entry.getKey();
            List<ClientOrder> componentOrders = entry.getValue();
            List<ClientShipment> componentShipments = clientShipments.stream()
                    .filter(ss -> componentOrders.stream().map(ClientOrder::getId).toList().contains(ss.getClientOrderId()))
                    .toList();

            int totalDeliveredComponentOrders = 0;
            float totalDelaysComponent = 0;
            float ratioOfOnTimeOrderDeliveriesComponent = 0;
            float ratioOfOnTimeShipmentDeliveriesComponent = 0;
            float averageShipmentsPerOrderComponent = 0;
            float averageTimeToShipOrderComponent = 0;

            float totalDeliveredQuantity = 0;
            float averageDeliveredQuantity = 0;
            float averageOrderQuantity = 0;
            float averageShipmentQuantity = 0;
            float deliveredPerOrderedRatio = 0;
            LocalDateTime firstDeliveryDate = componentOrders.stream()
                    .map(ClientOrder::getDeliveryDate).filter(Objects::nonNull) // Filter out null delivery dates
                    .min(LocalDateTime::compareTo).orElse(null);
            Map<Float, Float> deliveredQuantityOverTime = new HashMap<>();

            for (ClientOrder clientOrder : componentOrders) {
                // Compute delivery metrics
                if (clientOrder.getDeliveryDate() == null) continue;
                totalDeliveredComponentOrders++;

                if (clientOrder.getEstimatedDeliveryDate() != null) {
                    Duration orderDelay = Duration.between(clientOrder.getEstimatedDeliveryDate(), clientOrder.getDeliveryDate());
                    totalDelaysComponent += orderDelay.toDays();
                    if (totalDelaysComponent <= 0) {
                        ratioOfOnTimeOrderDeliveriesComponent++;
                    }
                }

                if (clientOrder.getOrderDate() != null) {
                    Duration shipDuration = Duration.between(clientOrder.getOrderDate(), clientOrder.getDeliveryDate());
                    averageTimeToShipOrderComponent += shipDuration.toDays();
                }

                List<ClientShipment> correspondingShipments = componentShipments.stream()
                        .filter(ss -> ss.getClientOrderId().equals(clientOrder.getId()))
                        .toList();

                averageShipmentsPerOrderComponent += correspondingShipments.size();

                // Compute quantity metrics
                totalDeliveredQuantity += clientOrder.getQuantity();
                averageDeliveredQuantity += clientOrder.getDeliveredQuantity();
                averageOrderQuantity += clientOrder.getQuantity();
                averageShipmentQuantity += correspondingShipments.stream().map(ClientShipment::getQuantity).reduce(0.0f, Float::sum);
                deliveredPerOrderedRatio += clientOrder.getDeliveredQuantity();
                if (firstDeliveryDate == null) continue;
                Duration timeFromFirstDelivery = Duration.between(firstDeliveryDate, clientOrder.getDeliveryDate());
                long days = timeFromFirstDelivery.toDays(); // This gives you the total days as a long
                deliveredQuantityOverTime.put((float) days, clientOrder.getDeliveredQuantity());
            }

            // Add to total
            totalDeliveredOrders += totalDeliveredComponentOrders;
            totalDelays += totalDelaysComponent;
            ratioOfOnTimeOrderDeliveries += ratioOfOnTimeOrderDeliveriesComponent;
            ratioOfOnTimeShipmentDeliveries += ratioOfOnTimeShipmentDeliveriesComponent;
            averageShipmentsPerOrder += averageShipmentsPerOrderComponent;
            averageTimeToShipOrder += averageTimeToShipOrderComponent;

            // Get component performance
            ComponentDeliveryPerformance componentPerformance = new ComponentDeliveryPerformance();
            componentPerformance.setComponentId(componentId);
            // TODO: Fix this
//            componentPerformance.setComponentName(componentOrders.getFirst().getComponent().getName());
            componentPerformance.setTotalDeliveredOrders(totalDeliveredComponentOrders);
            componentPerformance.setTotalDeliveredQuantity(totalDeliveredQuantity);
            if (totalDeliveredComponentOrders > 0) {
                componentPerformance.setAverageDeliveredQuantity(averageDeliveredQuantity / totalDeliveredComponentOrders);
                componentPerformance.setAverageOrderQuantity(averageOrderQuantity / totalDeliveredComponentOrders);
                componentPerformance.setDeliveredPerOrderedRatio(deliveredPerOrderedRatio / totalDeliveredComponentOrders);
                componentPerformance.setAverageShipmentQuantity(averageShipmentQuantity / totalDeliveredComponentOrders); // Not good yet
                componentPerformance.setFirstDeliveryDate(firstDeliveryDate);
                componentPerformance.setDeliveredQuantityOverTime(deliveredQuantityOverTime);
                componentPerformances.put(componentId, componentPerformance);
            }
        }

        // Calculate average metrics
        if (totalDeliveredOrders > 0) {
            report.setTotalDeliveredOrders(totalDeliveredOrders);
            report.setTotalDelays(totalDelays);
            report.setAverageDelayPerOrder(totalDelays / totalDeliveredOrders);
            report.setRatioOfOnTimeOrderDeliveries(ratioOfOnTimeOrderDeliveries / totalDeliveredOrders);
            report.setAverageDelayPerShipment(totalDelays / totalDeliveredOrders);
            report.setRatioOfOnTimeShipmentDeliveries(ratioOfOnTimeShipmentDeliveries / totalDeliveredOrders);
            report.setAverageShipmentsPerOrder(averageShipmentsPerOrder / totalDeliveredOrders);
            report.setAverageTimeToShipOrder(averageTimeToShipOrder / totalDeliveredOrders);
        }

        report.setComponentPerformances(componentPerformances);

        return report;
    }
}
