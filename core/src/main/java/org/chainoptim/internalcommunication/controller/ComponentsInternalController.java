package org.chainoptim.internalcommunication.controller;

import org.chainoptim.features.productpipeline.model.Component;
import org.chainoptim.features.productpipeline.repository.ComponentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequestMapping("/api/v1/internal/components")
public class ComponentsInternalController {

    private final ComponentRepository componentRepository;

    @Autowired
    public ComponentsInternalController(ComponentRepository componentRepository) {
        this.componentRepository = componentRepository;
    }

    @GetMapping("/{id}")
    public ResponseEntity<Component> getComponentById(@PathVariable Integer id) {
        Optional<Component> component = componentRepository.findById(id);
        return component.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }
}
