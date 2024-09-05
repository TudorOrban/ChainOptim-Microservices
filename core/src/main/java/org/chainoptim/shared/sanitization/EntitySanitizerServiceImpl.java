package org.chainoptim.shared.sanitization;

import org.chainoptim.core.settings.dto.CreateUserSettingsDTO;
import org.chainoptim.core.settings.dto.UpdateUserSettingsDTO;
import org.chainoptim.features.client.dto.*;
import org.chainoptim.features.factory.dto.CreateFactoryDTO;
import org.chainoptim.features.factory.dto.CreateFactoryInventoryItemDTO;
import org.chainoptim.features.factory.dto.UpdateFactoryDTO;
import org.chainoptim.features.factory.dto.UpdateFactoryInventoryItemDTO;
import org.chainoptim.features.product.dto.CreateProductDTO;
import org.chainoptim.features.product.dto.CreateUnitOfMeasurementDTO;
import org.chainoptim.features.product.dto.UpdateProductDTO;
import org.chainoptim.features.product.dto.UpdateUnitOfMeasurementDTO;
import org.chainoptim.features.productpipeline.dto.CreateComponentDTO;
import org.chainoptim.features.productpipeline.dto.CreateStageDTO;
import org.chainoptim.features.productpipeline.dto.UpdateComponentDTO;
import org.chainoptim.features.productpipeline.dto.UpdateStageDTO;
import org.chainoptim.shared.commonfeatures.location.dto.CreateLocationDTO;
import org.chainoptim.shared.commonfeatures.location.dto.UpdateLocationDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class EntitySanitizerServiceImpl implements EntitySanitizerService {

    private final SanitizationService sanitizationService;

    @Autowired
    public EntitySanitizerServiceImpl(SanitizationService sanitizationService) {
        this.sanitizationService = sanitizationService;
    }

    // Products
    public CreateProductDTO sanitizeCreateProductDTO(CreateProductDTO productDTO) {
        productDTO.setName(sanitizationService.sanitize(productDTO.getName()));
        productDTO.setDescription(sanitizationService.sanitize(productDTO.getDescription()));

        return productDTO;
    }

    public UpdateProductDTO sanitizeUpdateProductDTO(UpdateProductDTO productDTO) {
        productDTO.setName(sanitizationService.sanitize(productDTO.getName()));
        productDTO.setDescription(sanitizationService.sanitize(productDTO.getDescription()));

        return productDTO;
    }

    // Product pipeline
    public CreateStageDTO sanitizeCreateStageDTO(CreateStageDTO stageDTO) {
        stageDTO.setName(sanitizationService.sanitize(stageDTO.getName()));
        stageDTO.setDescription(sanitizationService.sanitize(stageDTO.getDescription()));

        return stageDTO;
    }

    public UpdateStageDTO sanitizeUpdateStageDTO(UpdateStageDTO stageDTO) {
        stageDTO.setName(sanitizationService.sanitize(stageDTO.getName()));
        stageDTO.setDescription(sanitizationService.sanitize(stageDTO.getDescription()));

        return stageDTO;
    }

    // Factories
    public CreateFactoryDTO sanitizeCreateFactoryDTO(CreateFactoryDTO factoryDTO) {
        factoryDTO.setName(sanitizationService.sanitize(factoryDTO.getName()));

        return factoryDTO;
    }

    public UpdateFactoryDTO sanitizeUpdateFactoryDTO(UpdateFactoryDTO factoryDTO) {
        factoryDTO.setName(sanitizationService.sanitize(factoryDTO.getName()));

        return factoryDTO;
    }

    public CreateFactoryInventoryItemDTO sanitizeCreateFactoryInventoryItemDTO(CreateFactoryInventoryItemDTO itemDTO) {
        return itemDTO;
    }

    public UpdateFactoryInventoryItemDTO sanitizeUpdateFactoryInventoryItemDTO(UpdateFactoryInventoryItemDTO itemDTO) {
        return itemDTO;
    }

    // Clients
    public CreateClientDTO sanitizeCreateClientDTO(CreateClientDTO clientDTO) {
        clientDTO.setName(sanitizationService.sanitize(clientDTO.getName()));

        return clientDTO;
    }

    public UpdateClientDTO sanitizeUpdateClientDTO(UpdateClientDTO clientDTO) {
        clientDTO.setName(sanitizationService.sanitize(clientDTO.getName()));

        return clientDTO;
    }

    public CreateClientOrderDTO sanitizeCreateClientOrderDTO(CreateClientOrderDTO orderDTO) {
        return orderDTO;
    }

    public UpdateClientOrderDTO sanitizeUpdateClientOrderDTO(UpdateClientOrderDTO orderDTO) {
        return orderDTO;
    }

    public CreateClientShipmentDTO sanitizeCreateClientShipmentDTO(CreateClientShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

    public UpdateClientShipmentDTO sanitizeUpdateClientShipmentDTO(UpdateClientShipmentDTO shipmentDTO) {
        return shipmentDTO;
    }

    // Locations
    public CreateLocationDTO sanitizeCreateLocationDTO(CreateLocationDTO locationDTO) {
        locationDTO.setAddress(sanitizationService.sanitize(locationDTO.getAddress()));
        locationDTO.setCity(sanitizationService.sanitize(locationDTO.getCity()));
        locationDTO.setState(sanitizationService.sanitize(locationDTO.getState()));
        locationDTO.setCountry(sanitizationService.sanitize(locationDTO.getCountry()));
        locationDTO.setZipCode(sanitizationService.sanitize(locationDTO.getZipCode()));

        return locationDTO;
    }

    public UpdateLocationDTO sanitizeUpdateLocationDTO(UpdateLocationDTO locationDTO) {
        locationDTO.setAddress(sanitizationService.sanitize(locationDTO.getAddress()));
        locationDTO.setCity(sanitizationService.sanitize(locationDTO.getCity()));
        locationDTO.setState(sanitizationService.sanitize(locationDTO.getState()));
        locationDTO.setCountry(sanitizationService.sanitize(locationDTO.getCountry()));
        locationDTO.setZipCode(sanitizationService.sanitize(locationDTO.getZipCode()));

        return locationDTO;
    }

    // Units of Measurement
    public CreateUnitOfMeasurementDTO sanitizeCreateUnitOfMeasurementDTO(CreateUnitOfMeasurementDTO unitOfMeasurementDTO) {
        unitOfMeasurementDTO.setName(sanitizationService.sanitize(unitOfMeasurementDTO.getName()));
        unitOfMeasurementDTO.setUnitType(sanitizationService.sanitize(unitOfMeasurementDTO.getUnitType()));

        return unitOfMeasurementDTO;
    }

    public UpdateUnitOfMeasurementDTO sanitizeUpdateUnitOfMeasurementDTO(UpdateUnitOfMeasurementDTO unitOfMeasurementDTO) {
        unitOfMeasurementDTO.setName(sanitizationService.sanitize(unitOfMeasurementDTO.getName()));
        unitOfMeasurementDTO.setUnitType(sanitizationService.sanitize(unitOfMeasurementDTO.getUnitType()));

        return unitOfMeasurementDTO;
    }

    // Component
    public CreateComponentDTO sanitizeCreateComponentDTO(CreateComponentDTO componentDTO) {
        componentDTO.setName(sanitizationService.sanitize(componentDTO.getName()));
        componentDTO.setDescription(sanitizationService.sanitize(componentDTO.getDescription()));

        return componentDTO;
    }

    public UpdateComponentDTO sanitizeUpdateComponentDTO(UpdateComponentDTO componentDTO) {
        componentDTO.setName(sanitizationService.sanitize(componentDTO.getName()));
        componentDTO.setDescription(sanitizationService.sanitize(componentDTO.getDescription()));

        return componentDTO;
    }

    // Settings
    public CreateUserSettingsDTO sanitizeCreateUserSettingsDTO(CreateUserSettingsDTO userSettingsDTO) {
        return userSettingsDTO;
    }

    public UpdateUserSettingsDTO sanitizeUpdateUserSettingsDTO(UpdateUserSettingsDTO userSettingsDTO) {
        return userSettingsDTO;
    }
}
